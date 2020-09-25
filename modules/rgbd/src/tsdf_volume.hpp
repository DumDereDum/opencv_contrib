// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_RGBD_TSDF_VOLUME_H__
#define __OPENCV_RGBD_TSDF_VOLUME_H__

//#include "kinfu_frame.hpp"
#include "tsdf.hpp"

namespace cv
{
namespace kinfu
{

typedef int8_t TsdfType;
typedef uchar WeightType;
/*
struct TsdfVoxel
{
    TsdfType tsdf;
    WeightType weight;
};
*/

//typedef Vec<uchar, sizeof(TsdfVoxel)> VecTsdfVoxel;

class CV_EXPORTS_W NewVolume
{
    public:
        NewVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
            /*TSDFVolume*/ float _truncDist, int _maxWeight, Point3i _resolution, bool zFirstMemOrder = true);

        virtual ~NewVolume() {};

        void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
            const cv::kinfu::Intr& intrinsics);
        /*
        virtual void raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
            cv::Size frameSize, cv::OutputArray points,
            cv::OutputArray normals) const = 0;
        virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0;
        virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;
        */

        void reset();
        
        TsdfVoxel at(const cv::Vec3i& volumeIdx) const;


    public:
        // Volume
        const float voxelSize;
        const float voxelSizeInv;
        const cv::Affine3f pose;
        const float raycastStepFactor;

        // TSDF Volume
        Point3i volResolution;
        WeightType maxWeight;

        Point3f volSize;
        float truncDist;
        Vec4i volDims;
        Vec8i neighbourCoords;

        // TSDF Volume CPU
        Vec6f frameParams;
        Mat pixNorms;
        // See zFirstMemOrder arg of parent class constructor
        // for the array layout info
        // Consist of Voxel elements
        Mat volume;
};

}  // namespace kinfu
}  // namespace cv
#endif

/*
TSDFVolume::TSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                       int _maxWeight, Point3i _resolution, bool zFirstMemOrder)
    : Volume(_voxelSize, _pose, _raycastStepFactor),
      volResolution(_resolution),
      maxWeight( WeightType(_maxWeight) )

      TSDFVolumeCPU::TSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                             float _truncDist, int _maxWeight, Vec3i _resolution,
                             bool zFirstMemOrder)
    : TSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _resolution,
                 zFirstMemOrder)
{
    volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<TsdfVoxel>());

    reset();
}
*/
