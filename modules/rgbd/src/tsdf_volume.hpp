// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_RGBD_TSDF_VOLUME_H__
#define __OPENCV_RGBD_TSDF_VOLUME_H__

namespace cv
{
namespace kinfu
{

class CV_EXPORTS_W NewVolume
{
public:
    NewVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor)
        : voxelSize(_voxelSize),
        voxelSizeInv(1.0f / voxelSize),
        pose(_pose),
        raycastStepFactor(_raycastStepFactor)
    {
    }

    virtual ~NewVolume() {};

    virtual void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
        const cv::kinfu::Intr& intrinsics) = 0;
    virtual void raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
        cv::Size frameSize, cv::OutputArray points,
        cv::OutputArray normals) const = 0;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;
    virtual void reset() = 0;

public:
    const float voxelSize;
    const float voxelSizeInv;
    const cv::Affine3f pose;
    const float raycastStepFactor;
};

}  // namespace kinfu
}  // namespace cv
#endif
