// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "tsdf_functions.hpp"

namespace cv {

namespace kinfu {

cv::Mat preCalculationPixNorm(Depth depth, const Intr& intrinsics)
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

const bool fixMissingData = false;
depthType bilinearDepth(const Depth& m, cv::Point2f pt)
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

void integrateVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::kinfu::Intr& intrinsics, InputArray _pixNorms, InputArray _volume)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    cv::Affine3f vpose(_pose);
    Depth depth = _depth.getMat();

    Range integrateRange(0, volResolution.x);

    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();
    const Intr::Projector proj(intrinsics.makeProjector());
    const cv::Affine3f vol2cam(Affine3f(cameraPose.inv()) * vpose);
    const float truncDistInv(1.f / truncDist);
    const float dfac(1.f / depthFactor);
    TsdfVoxel* volDataStart = volume.ptr<TsdfVoxel>();;

#if USE_INTRINSICS
    auto IntegrateInvoker = [&](const Range& range)
    {
        // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
        Point3f zStepPt = Point3f(vol2cam.matrix(0, 2),
            vol2cam.matrix(1, 2),
            vol2cam.matrix(2, 2)) * voxelSize;

        v_float32x4 zStep(zStepPt.x, zStepPt.y, zStepPt.z, 0);
        v_float32x4 vfxy(proj.fx, proj.fy, 0.f, 0.f), vcxy(proj.cx, proj.cy, 0.f, 0.f);
        const v_float32x4 upLimits = v_cvt_f32(v_int32x4(depth.cols - 1, depth.rows - 1, 0, 0));

        for (int x = range.start; x < range.end; x++)
        {
            TsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                TsdfVoxel* volDataY = volDataX + y * volStrides[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam * (Point3f((float)x, (float)y, 0) * voxelSize);
                v_float32x4 camSpacePt(basePt.x, basePt.y, basePt.z, 0);

                int startZ, endZ;
                if (abs(zStepPt.z) > 1e-5)
                {
                    int baseZ = (int)(-basePt.z / zStepPt.z);
                    if (zStepPt.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volResolution.z;
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
                        endZ = volResolution.z;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(int(volResolution.z), endZ);
                for (int z = startZ; z < endZ; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;
                    camSpacePt += zStep;

                    float zCamSpace = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(camSpacePt))).get0();
                    if (zCamSpace <= 0.f)
                        continue;

                    v_float32x4 camPixVec = camSpacePt / v_setall_f32(zCamSpace);
                    v_float32x4 projected = v_muladd(camPixVec, vfxy, vcxy);
                    // leave only first 2 lanes
                    projected = v_reinterpret_as_f32(v_reinterpret_as_u32(projected) &
                        v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0));

                    depthType v;
                    // bilinearly interpolate depth at projected
                    {
                        const v_float32x4& pt = projected;
                        // check coords >= 0 and < imgSize
                        v_uint32x4 limits = v_reinterpret_as_u32(pt < v_setzero_f32()) |
                            v_reinterpret_as_u32(pt >= upLimits);
                        limits = limits | v_rotate_right<1>(limits);
                        if (limits.get0())
                            continue;

                        // xi, yi = floor(pt)
                        v_int32x4 ip = v_floor(pt);
                        v_int32x4 ipshift = ip;
                        int xi = ipshift.get0();
                        ipshift = v_rotate_right<1>(ipshift);
                        int yi = ipshift.get0();

                        const depthType* row0 = depth[yi + 0];
                        const depthType* row1 = depth[yi + 1];

                        // v001 = [v(xi + 0, yi + 0), v(xi + 1, yi + 0)]
                        v_float32x4 v001 = v_load_low(row0 + xi);
                        // v101 = [v(xi + 0, yi + 1), v(xi + 1, yi + 1)]
                        v_float32x4 v101 = v_load_low(row1 + xi);

                        v_float32x4 vall = v_combine_low(v001, v101);

                        // assume correct depth is positive
                        // don't fix missing data
                        if (v_check_all(vall > v_setzero_f32()))
                        {
                            v_float32x4 t = pt - v_cvt_f32(ip);
                            float tx = t.get0();
                            t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
                            v_float32x4 ty = v_setall_f32(t.get0());
                            // vx is y-interpolated between rows 0 and 1
                            v_float32x4 vx = v001 + ty * (v101 - v001);
                            float v0 = vx.get0();
                            vx = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vx)));
                            float v1 = vx.get0();
                            v = v0 + tx * (v1 - v0);
                        }
                        else
                            continue;
                    }

                    // norm(camPixVec) produces double which is too slow
                    int _u = (int)projected.get0();
                    int _v = (int)v_rotate_right<1>(projected).get0();
                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;
                    float pixNorm = pixNorms.at<float>(_v, _u);
                    // float pixNorm = sqrt(v_reduce_sum(camPixVec*camPixVec));
                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - zCamSpace);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        TsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = (weight + 1) < maxWeight ? (weight + 1) : (WeightType) maxWeight;
                    }
                }
            }
        }
    };
#else
    auto IntegrateInvoker = [&](const Range& range)
    {
        for (int x = range.start; x < range.end; x++)
        {
            TsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                TsdfVoxel* volDataY = volDataX + y * volStrides[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam * (Point3f(float(x), float(y), 0.0f) * voxelSize);
                Point3f camSpacePt = basePt;
                // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                // zStep == vol2cam*[Point3f(x, y, 1) - Point3f(x, y, 0)]*voxelSize
                Point3f zStep = Point3f(vol2cam.matrix(0, 2),
                    vol2cam.matrix(1, 2),
                    vol2cam.matrix(2, 2)) * voxelSize;
                int startZ, endZ;
                if (abs(zStep.z) > 1e-5)
                {
                    int baseZ = int(-basePt.z / zStep.z);
                    if (zStep.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volResolution.z;
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
                        endZ = volResolution.z;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(int(volResolution.z), endZ);

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
                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;
                    float pixNorm = pixNorms.at<float>(_v, _u);

                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        TsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = min(int(weight + 1), int(maxWeight));
                    }
                }
            }
        }
    };
#endif

    parallel_for_(integrateRange, IntegrateInvoker);

}

size_t calc_hash(Vec4i x)
{
    uint32_t seed = 0;
    constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
    //uint32_t GOLDEN_RATIO = 0x9e3779b9;
    //seed ^= x[0] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    //seed ^= x[1] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    //seed ^= x[2] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    //std::cout << " lol "  << x[0]<<std::endl;
    for (int i = 0; i < 3; i++)
    {
        //seed ^= std::hash<int>()(x[i]) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        seed ^= x[i] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        //std::cout << x[i] << "|" << seed << std::endl;
    }
    return seed;
}

VolumesTable::VolumesTable()
{
    this->volumes = cv::Mat(hash_divisor * list_size, 1, rawType<Volume_NODE>());
    for (int i = 0; i < volumes.size().height; i++)
    {
        Volume_NODE& v = volumes.at<Volume_NODE>(i, 0);
        v.idx = nan4;
        v.row = -1;
        v.nextVolumeRow = -1;
        v.isActive = 0;
        v.lastVisibleIndex = -1;
        //v.tmp = i;
    }
}

void VolumesTable::update(Vec3i indx)
{
    Vec4i idx(indx[0], indx[1], indx[2], 0);
    int hash = int(calc_hash(idx) % hash_divisor);
    int num  = 1;
    int start = hash * num * list_size;
    int i = start;

    while (i != -1)
    {
        Volume_NODE& v = volumes.at<Volume_NODE>(i, 0);
        if (v.idx == idx)
            return;
        //find nan cheking for int or Vec3i
        //if (isNaN(Point3i(v.idx)))
        if (v.idx[0] == -2147483647)
        {
            v.idx = idx;
            v.nextVolumeRow = getNextVolume(hash, num, i, start);
            indexes.push_back(indx);
            indexesGPU.push_back(idx);
            return;
        }
        i = v.nextVolumeRow;
    }
}

void VolumesTable::update(Vec3i indx, int row)
{
    Vec4i idx(indx[0], indx[1], indx[2], 0);
    int hash = int(calc_hash(idx) % hash_divisor);
    int num = 1;
    int start = hash * num * list_size;
    int i = start;

    while (i != -1)
    {
        Volume_NODE& v = volumes.at<Volume_NODE>(i, 0);
        if (v.idx == idx)
        {
            v.row = row;
            return;
        }
        //find nan cheking for int or Vec3i
        //if (isNaN(Point3i(v.idx)))
        if (v.idx[0] == -2147483647)
        {
            v.idx = idx;
            v.row = row;
            v.nextVolumeRow = getNextVolume(hash, num, i, start);
            indexes.push_back(indx);
            indexesGPU.push_back(idx);
            return;
        }
        i = v.nextVolumeRow;
    }
}

void VolumesTable::update(Vec3i indx, int isActive, int lastVisibleIndex)
{
    Vec4i idx(indx[0], indx[1], indx[2], 0);
    int hash = int(calc_hash(idx) % hash_divisor);
    int num = 1;
    int start = hash * num * list_size;
    int i = start;

    while (i != -1)
    {
        Volume_NODE& v = volumes.at<Volume_NODE>(i, 0);
        if (v.idx == idx)
        {
            v.isActive = isActive;
            v.lastVisibleIndex = lastVisibleIndex;
            return;
        }
        //find nan cheking for int or Vec3i
        //if (isNaN(Point3i(v.idx)))
        if (v.idx[0] == -2147483647)
        {
            v.idx = idx;
            v.nextVolumeRow = getNextVolume(hash, num, i, start);
            v.isActive = isActive;
            v.lastVisibleIndex = lastVisibleIndex;
            indexes.push_back(indx);
            indexesGPU.push_back(idx);
            return;
        }
        i = v.nextVolumeRow;
    }
}

void VolumesTable::updateActive(Vec3i indx, int isActive)
{
    Vec4i idx(indx[0], indx[1], indx[2], 0);
    int hash = int(calc_hash(idx) % hash_divisor);
    int num = 1;
    int start = hash * num * list_size;
    int i = start;

    while (i != -1)
    {
        Volume_NODE& v = volumes.at<Volume_NODE>(i, 0);
        if (v.idx == idx)
        {
            v.isActive = isActive;
            return;
        }
        //find nan cheking for int or Vec3i
        //if (isNaN(Point3i(v.idx)))
        if (v.idx[0] == -2147483647)
        {
            return;
        }
        i = v.nextVolumeRow;
    }
}

bool VolumesTable::getActive(Vec3i indx) const
{
    Vec4i idx(indx[0], indx[1], indx[2], 0);
    int hash = int(calc_hash(idx) % hash_divisor);
    int num = 1;
    int i = hash * num * list_size;
    while (i != -1)
    {
        Volume_NODE v = volumes.at<Volume_NODE>(i, 0);
        if (v.idx == idx)
            return bool(v.isActive);
        if (v.idx[0] == -2147483647)
            return false;
        i = v.nextVolumeRow;
    }
    return false;
}

int VolumesTable::getNextVolume(int hash, int& num, int i, int start)
{
    if (i != start && i % list_size == 0)
    {
        if (num < bufferNums)
        {
            num++;
        }
        else
        {
            this->expand();
            num++;
        }
        return hash * num * list_size;
    }
    else
    {
        return i+1;
    }
}

void VolumesTable::expand()
{
    this->volumes.resize(hash_divisor * (bufferNums + 1));
    this->bufferNums++;
}

int VolumesTable::find_Volume(Vec3i indx) const
{
    Vec4i idx(indx[0], indx[1], indx[2], 0);
    //std::cout << "find_Volume -> ";
    int hash = int(calc_hash(idx) % hash_divisor);
    int num = 1;
    int i = hash * num * list_size;
    //std::cout <<"[ "<< idx<<" ]= " << calc_hash(idx) <<" = "<< hash << std::endl;
    while (i != -1)
    {
        Volume_NODE v = volumes.at<Volume_NODE>(i, 0);
        //std::cout << " | i = " << i << " idx=" << v.idx << " row=" << v.row << " next=" << v.nextVolumeRow << std::endl;
        if (v.idx == idx)
            return v.row;
        //find nan cheking for int or Vec3i
        //if (isNaN(Point3i(v.idx)))
        if (v.idx[0] == -2147483647)
            return -2;
        i = v.nextVolumeRow;
    }
    return -2;
}
bool VolumesTable::isExist(Vec3i indx)
{
    //std::cout << "isExist -> ";
    if (this->find_Volume(indx) == -2)
        return false;
    return true;
}


} // namespace kinfu
} // namespace cv
