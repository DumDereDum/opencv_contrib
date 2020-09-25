// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "tsdf_volume.hpp"
#include "kinfu_frame.hpp"

namespace cv
{
namespace kinfu
{
	NewVolume::NewVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
        /*TSDFVolume*/   float _truncDist, int _maxWeight, Point3i _resolution, bool zFirstMemOrder)
        : voxelSize(_voxelSize),
        voxelSizeInv(1.0f / voxelSize),
        pose(_pose),
        raycastStepFactor(_raycastStepFactor),
        volResolution(_resolution),
        maxWeight(WeightType(_maxWeight))
	{
        //TSDF Volume
        CV_Assert(_maxWeight < 255);
        // Unlike original code, this should work with any volume size
        // Not only when (x,y,z % 32) == 0
        volSize = Point3f(volResolution) * voxelSize;
        truncDist = std::max(_truncDist, 2.1f * voxelSize);

        // (xRes*yRes*zRes) array
        // Depending on zFirstMemOrder arg:
        // &elem(x, y, z) = data + x*zRes*yRes + y*zRes + z;
        // &elem(x, y, z) = data + x + y*xRes + z*xRes*yRes;
        int xdim, ydim, zdim;
        if (zFirstMemOrder)
        {
            xdim = volResolution.z * volResolution.y;
            ydim = volResolution.z;
            zdim = 1;
        }
        else
        {
            xdim = 1;
            ydim = volResolution.x;
            zdim = volResolution.x * volResolution.y;
        }

        volDims = Vec4i(xdim, ydim, zdim);
        neighbourCoords = Vec8i(
            volDims.dot(Vec4i(0, 0, 0)),
            volDims.dot(Vec4i(0, 0, 1)),
            volDims.dot(Vec4i(0, 1, 0)),
            volDims.dot(Vec4i(0, 1, 1)),
            volDims.dot(Vec4i(1, 0, 0)),
            volDims.dot(Vec4i(1, 0, 1)),
            volDims.dot(Vec4i(1, 1, 0)),
            volDims.dot(Vec4i(1, 1, 1))
        );

        // TSDF Volume CPU
        volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<TsdfVoxel>());
        //std::cout << 1;
        reset();

	}

    static inline v_float32x4 tsdfToFloat_INTR(const v_int32x4& num)
    {
        v_float32x4 num128 = v_setall_f32(-1.f / 128.f);
        return v_cvt_f32(num) * num128;
    }

    static inline TsdfType floatToTsdf(float num)
    {
        //CV_Assert(-1 < num <= 1);
        int8_t res = int8_t(num * (-128.f));
        res = res ? res : (num < 0 ? 1 : -1);
        return res;
    }

    static inline float tsdfToFloat(TsdfType num)
    {
        return float(num) * (-1.f / 128.f);
    }

    static const bool fixMissingData = false;

    static inline depthType bilinearDepth(const Depth& m, cv::Point2f pt)
    {
        const depthType defaultValue = qnan;
        if (pt.x < 0 || pt.x >= m.cols - 1 ||
            pt.y < 0 || pt.y >= m.rows - 1)
            return defaultValue;

        int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

        const depthType* row0 = m[yi + 0];
        const depthType* row1 = m[yi + 1];

        depthType v00 = row0[xi + 0];
        depthType v01 = row0[xi + 1];
        depthType v10 = row1[xi + 0];
        depthType v11 = row1[xi + 1];

        // assume correct depth is positive
        bool b00 = v00 > 0;
        bool b01 = v01 > 0;
        bool b10 = v10 > 0;
        bool b11 = v11 > 0;

        if (!fixMissingData)
        {
            if (!(b00 && b01 && b10 && b11))
                return defaultValue;
            else
            {
                float tx = pt.x - xi, ty = pt.y - yi;
                depthType v0 = v00 + tx * (v01 - v00);
                depthType v1 = v10 + tx * (v11 - v10);
                return v0 + ty * (v1 - v0);
            }
        }
        else
        {
            int nz = b00 + b01 + b10 + b11;
            if (nz == 0)
            {
                return defaultValue;
            }
            if (nz == 1)
            {
                if (b00) return v00;
                if (b01) return v01;
                if (b10) return v10;
                if (b11) return v11;
            }
            if (nz == 2)
            {
                if (b00 && b10) v01 = v00, v11 = v10;
                if (b01 && b11) v00 = v01, v10 = v11;
                if (b00 && b01) v10 = v00, v11 = v01;
                if (b10 && b11) v00 = v10, v01 = v11;
                if (b00 && b11) v01 = v10 = (v00 + v11) * 0.5f;
                if (b01 && b10) v00 = v11 = (v01 + v10) * 0.5f;
            }
            if (nz == 3)
            {
                if (!b00) v00 = v10 + v01 - v11;
                if (!b01) v01 = v00 + v11 - v10;
                if (!b10) v10 = v00 + v11 - v01;
                if (!b11) v11 = v01 + v10 - v00;
            }

            float tx = pt.x - xi, ty = pt.y - yi;
            depthType v0 = v00 + tx * (v01 - v00);
            depthType v1 = v10 + tx * (v11 - v10);
            return v0 + ty * (v1 - v0);
        }
    }

    struct IntegrateInvoker : ParallelLoopBody
    {
        IntegrateInvoker(NewVolume& _volume, const Depth& _depth, const Intr& intrinsics,
            const cv::Matx44f& cameraPose, float depthFactor, Mat _pixNorms) :
            ParallelLoopBody(),
            volume(_volume),
            depth(_depth),
            intr(intrinsics),
            proj(intrinsics.makeProjector()),
            vol2cam(Affine3f(cameraPose.inv())* _volume.pose),
            truncDistInv(1.f / _volume.truncDist),
            dfac(1.f / depthFactor),
            pixNorms(_pixNorms)
        {
        std::cout << "i";
            volDataStart = volume.volume.ptr<TsdfVoxel>();
        }

        virtual void operator() (const Range& range) const override
        {
            std::cout << "3";
            for (int x = range.start; x < range.end; x++)
            {
                TsdfVoxel* volDataX = volDataStart + x * volume.volDims[0];
                for (int y = 0; y < volume.volResolution.y; y++)
                {
                    TsdfVoxel* volDataY = volDataX + y * volume.volDims[1];
                    // optimization of camSpace transformation (vector addition instead of matmul at each z)
                    Point3f basePt = vol2cam * (Point3f(float(x), float(y), 0.0f) * volume.voxelSize);
                    Point3f camSpacePt = basePt;
                    // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                    // zStep == vol2cam*[Point3f(x, y, 1) - Point3f(x, y, 0)]*voxelSize
                    Point3f zStep = Point3f(vol2cam.matrix(0, 2),
                        vol2cam.matrix(1, 2),
                        vol2cam.matrix(2, 2)) * volume.voxelSize;
                    int startZ, endZ;
                    if (abs(zStep.z) > 1e-5)
                    {
                        int baseZ = int(-basePt.z / zStep.z);
                        if (zStep.z > 0)
                        {
                            startZ = baseZ;
                            endZ = volume.volResolution.z;
                        }
                        else
                        {
                            startZ = 0;
                            endZ = baseZ;
                        }
                    }
                    else
                    {
                        if (basePt.z > 0)
                        {
                            startZ = 0;
                            endZ = volume.volResolution.z;
                        }
                        else
                        {
                            // z loop shouldn't be performed
                            startZ = endZ = 0;
                        }
                    }
                    startZ = max(0, startZ);
                    endZ = min(volume.volResolution.z, endZ);

                    for (int z = startZ; z < endZ; z++)
                    {
                        // optimization of the following:
                        //Point3f volPt = Point3f(x, y, z)*volume.voxelSize;
                        //Point3f camSpacePt = vol2cam * volPt;

                        camSpacePt += zStep;
                        if (camSpacePt.z <= 0)
                            continue;

                        Point3f camPixVec;
                        Point2f projected = proj(camSpacePt, camPixVec);

                        depthType v = bilinearDepth(depth, projected);
                        if (v == 0) {
                            continue;
                        }

                        int _u = projected.x;
                        int _v = projected.y;
                        if (!(_u >= 0 && _u < depth.rows && _v >= 0 && _v < depth.cols))
                            continue;
                        float pixNorm = pixNorms.at<float>(_u, _v);

                        // difference between distances of point and of surface to camera
                        float sdf = pixNorm * (v * dfac - camSpacePt.z);
                        // possible alternative is:
                        // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                        std::cout << "sdf: " << sdf << std::endl;
                        if (sdf >= -volume.truncDist)
                        {
                            TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                            TsdfVoxel& voxel = volDataY[z * volume.volDims[2]];
                            WeightType& weight = voxel.weight;
                            TsdfType& value = voxel.tsdf;
                            std::cout << value << std::endl;
                            // update TSDF
                            value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                            weight = min(int(weight + 1), int(volume.maxWeight));
                        }
                    }
                }
            }
        }

        NewVolume& volume;
        const Depth& depth;
        const Intr& intr;
        const Intr::Projector proj;
        const cv::Affine3f vol2cam;
        const float truncDistInv;
        const float dfac;
        TsdfVoxel* volDataStart;
        Mat pixNorms;
    };

    static cv::Mat preCalculationPixNorm(Depth depth, const Intr& intrinsics)
    {
        int height = depth.rows;
        int widht = depth.cols;
        Point2f fl(intrinsics.fx, intrinsics.fy);
        Point2f pp(intrinsics.cx, intrinsics.cy);
        Mat pixNorm(height, widht, CV_32F);
        std::vector<float> x(widht);
        std::vector<float> y(height);
        for (int i = 0; i < widht; i++)
            x[i] = (i - pp.x) / fl.x;
        for (int i = 0; i < height; i++)
            y[i] = (i - pp.y) / fl.y;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < widht; j++)
            {
                pixNorm.at<float>(i, j) = sqrtf(x[j] * x[j] + y[i] * y[i] + 1.0f);
            }
        }
        return pixNorm;
    }

    // use depth instead of distance (optimization)
    void NewVolume::integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
        const cv::kinfu::Intr& intrinsics)
    {
        std::cout << "integrate" << std::endl;
        CV_TRACE_FUNCTION();

        CV_Assert(_depth.type() == DEPTH_TYPE);
        CV_Assert(!_depth.empty());
        Depth depth = _depth.getMat();
        if (!(frameParams[0] == depth.rows && frameParams[1] == depth.cols &&
            frameParams[2] == intrinsics.fx && frameParams[3] == intrinsics.fy &&
            frameParams[4] == intrinsics.cx && frameParams[5] == intrinsics.cy))
        {
            frameParams[0] = (float)depth.rows; frameParams[1] = (float)depth.cols;
            frameParams[2] = intrinsics.fx;     frameParams[3] = intrinsics.fy;
            frameParams[4] = intrinsics.cx;     frameParams[5] = intrinsics.cy;

            pixNorms = preCalculationPixNorm(depth, intrinsics);
        }
        std::cout << 1;
        //IntegrateInvoker ii(*this, depth, intrinsics, cameraPose, depthFactor, pixNorms);
        Range range(0, volResolution.x);
        //parallel_for_(range, ii);
        //ii(range);
        //std::cout << 2;

        NewVolume volume(*this);
        const Intr& intr(intrinsics);
        const Intr::Projector proj(intrinsics.makeProjector());
        const cv::Affine3f vol2cam(Affine3f(cameraPose.inv()) * volume.pose);
        const float truncDistInv(1.f / volume.truncDist);
        const float dfac(1.f / depthFactor);
        TsdfVoxel* volDataStart = volume.volume.ptr<TsdfVoxel>();;


        auto _IntegrateInvoker = [&](const Range& range)
        {
            std::cout << "3";
            for (int x = range.start; x < range.end; x++)
            {
                TsdfVoxel* volDataX = volDataStart + x * volume.volDims[0];
                for (int y = 0; y < volume.volResolution.y; y++)
                {
                    TsdfVoxel* volDataY = volDataX + y * volume.volDims[1];
                    // optimization of camSpace transformation (vector addition instead of matmul at each z)
                    Point3f basePt = vol2cam * (Point3f(float(x), float(y), 0.0f) * volume.voxelSize);
                    Point3f camSpacePt = basePt;
                    // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                    // zStep == vol2cam*[Point3f(x, y, 1) - Point3f(x, y, 0)]*voxelSize
                    Point3f zStep = Point3f(vol2cam.matrix(0, 2),
                        vol2cam.matrix(1, 2),
                        vol2cam.matrix(2, 2)) * volume.voxelSize;
                    int startZ, endZ;
                    if (abs(zStep.z) > 1e-5)
                    {
                        int baseZ = int(-basePt.z / zStep.z);
                        if (zStep.z > 0)
                        {
                            startZ = baseZ;
                            endZ = volume.volResolution.z;
                        }
                        else
                        {
                            startZ = 0;
                            endZ = baseZ;
                        }
                    }
                    else
                    {
                        if (basePt.z > 0)
                        {
                            startZ = 0;
                            endZ = volume.volResolution.z;
                        }
                        else
                        {
                            // z loop shouldn't be performed
                            startZ = endZ = 0;
                        }
                    }
                    startZ = max(0, startZ);
                    endZ = min(volume.volResolution.z, endZ);

                    for (int z = startZ; z < endZ; z++)
                    {
                        // optimization of the following:
                        //Point3f volPt = Point3f(x, y, z)*volume.voxelSize;
                        //Point3f camSpacePt = vol2cam * volPt;

                        camSpacePt += zStep;
                        if (camSpacePt.z <= 0)
                            continue;

                        Point3f camPixVec;
                        Point2f projected = proj(camSpacePt, camPixVec);

                        depthType v = bilinearDepth(depth, projected);
                        if (v == 0) {
                            continue;
                        }

                        int _u = projected.x;
                        int _v = projected.y;
                        if (!(_u >= 0 && _u < depth.rows && _v >= 0 && _v < depth.cols))
                            continue;
                        float pixNorm = pixNorms.at<float>(_u, _v);

                        // difference between distances of point and of surface to camera
                        float sdf = pixNorm * (v * dfac - camSpacePt.z);
                        // possible alternative is:
                        // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                        std::cout << "sdf: " << sdf << std::endl;
                        if (sdf >= -volume.truncDist)
                        {
                            TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                            TsdfVoxel& voxel = volDataY[z * volume.volDims[2]];
                            WeightType& weight = voxel.weight;
                            TsdfType& value = voxel.tsdf;
                            std::cout << value << std::endl;
                            // update TSDF
                            value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                            weight = min(int(weight + 1), int(volume.maxWeight));
                        }
                    }
                }
            }
        };
        _IntegrateInvoker(range);
        std::cout << 2;
    }

    void NewVolume::reset()
    {
        CV_TRACE_FUNCTION();

        volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
    }

    TsdfVoxel NewVolume::at(const cv::Vec3i& volumeIdx) const
    {
        //! Out of bounds
        if ((volumeIdx[0] >= volResolution.x || volumeIdx[0] < 0) ||
            (volumeIdx[1] >= volResolution.y || volumeIdx[1] < 0) ||
            (volumeIdx[2] >= volResolution.z || volumeIdx[2] < 0))
        {
            TsdfVoxel dummy;
            dummy.tsdf = floatToTsdf(1.0f);
            dummy.weight = 0;
            return dummy;
        }

        const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();
        int coordBase =
            volumeIdx[0] * volDims[0] + volumeIdx[1] * volDims[1] + volumeIdx[2] * volDims[2];
        return volData[coordBase];
    }

}  // namespace kinfu
}  // namespace cv
