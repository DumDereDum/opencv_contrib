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

class CV_EXPORTS_W _NewVolume
{
    public:
        _NewVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
            /*TSDFVolume*/ float _truncDist, int _maxWeight, Point3i _resolution, bool zFirstMemOrder = true);

        virtual ~_NewVolume() {};

        void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
            const cv::kinfu::Intr& intrinsics, InputArray pixNorms, InputArray _volume);

        void reset();
        
        TsdfVoxel at(const cv::Vec3i& volumeIdx, InputArray _volume) const;


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
        //Vec6f frameParams;
        //Mat pixNorms;
        // See zFirstMemOrder arg of parent class constructor
        // for the array layout info
        // Consist of Voxel elements
        //Mat volume;
};

}  // namespace kinfu
}  // namespace cv
#endif
