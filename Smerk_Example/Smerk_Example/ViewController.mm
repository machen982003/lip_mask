//
//  ViewController.m
//  Smerk_Example
//
//  Created by Simon Lucey on 11/30/15.
//  Copyright © 2015 CMU_16432. All rights reserved.
//

#import "ViewController.h"
#import "SMKDetectionCamera.h"
#import <GPUImage/GPUImage.h>

#ifdef __cplusplus
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#endif

#define PI 3.1415926

using namespace std;
@interface ViewController () {
    // Setup the view (this time using GPUImageView)
    GPUImageView *cameraView_;
    SMKDetectionCamera *detector_; // Detector that should be used
    UIView *faceFeatureTrackingView_; // View for showing bounding box around the face
    UIImageView *contourmouth_;
    CGAffineTransform cameraOutputToPreviewFrameTransform_;
    CGAffineTransform portraitRotationTransform_;
    CGAffineTransform texelToPixelTransform_;
    GPUImageRawDataOutput *rawDataOutput; //can process the gpu data
    CGRect mouth_rect;
    cv::Mat mouth_mask;
    int upb ;
    int btb ;
    BOOL have_mouth;
    BOOL finished_processing;
     // save the gpu image

}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    [self.view.layer setAffineTransform:CGAffineTransformMakeScale(-1, 1)];
    // Setup GPUImageView (not we are not using UIImageView here).........
    cameraView_ = [[GPUImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
    
    // Set the face detector to be used
    detector_ = [[SMKDetectionCamera alloc] initWithSessionPreset:AVCaptureSessionPreset640x480 cameraPosition:AVCaptureDevicePositionFront];
   // [self.view addSubview:resultView_];
    rawDataOutput = [[GPUImageRawDataOutput alloc] initWithImageSize:CGSizeMake(480,640) resultsInBGRAFormat:true];
    [detector_ setOutputImageOrientation:UIInterfaceOrientationPortrait]; // Set to portrait
    cameraView_.fillMode = kGPUImageFillModePreserveAspectRatio;

    GPUImageMedianFilter *medianFilter = [[GPUImageMedianFilter alloc] init];
    have_mouth = false;
    finished_processing = true;
    mouth_rect = CGRectMake(0, 0, 0, 0);
    cout<<sizeof(GLubyte)<<endl;
    [detector_ addTarget:cameraView_];
    [detector_ addTarget:medianFilter];
    [medianFilter addTarget:rawDataOutput];

    // Important: add as a subview
    [self.view addSubview:cameraView_];
    // Setup the face box view
    [self setupFaceTrackingViews];
    [self setupContourMouth];
    [self calculateTransformations];
    
    // Set the block for running face detector

    [detector_ beginDetecting:kFaceFeatures | kMachineAndFaceMetaData
                        codeTypes:@[AVMetadataObjectTypeQRCode]
               withDetectionBlock:^(SMKDetectionOptions detectionType, NSArray *detectedObjects, CGRect clapOrRectZero) {
                   // Check if the kFaceFeatures have been discovered
                   if (detectionType & kFaceFeatures) {
                       [self updateFaceFeatureTrackingViewWithObjects:detectedObjects];
                    }
               }];
    //set up the contour extraction method
    
    __unsafe_unretained ViewController *weakself = self;

    [rawDataOutput setNewFrameAvailableBlock:^{

        [weakself->rawDataOutput lockFramebufferForReading];
     
        GLubyte *outputByte = weakself->rawDataOutput.rawBytesForImage;
        if (have_mouth == true && finished_processing == true) {
            [weakself extract_mout:outputByte];
        }
        [weakself->rawDataOutput unlockFramebufferAfterReading];
    }];

    // Finally start the camera
    [detector_ startCameraCapture];
}

-(void)extract_mout:(GLubyte *) outputByte
{
    finished_processing = false;
    int col_num = floor(480);
    int row_nun = floor(640);
    cv::Mat mout_rgb_raw(row_nun,col_num,CV_8UC4,outputByte);
    cv::Mat mout_rgb;
    mout_rgb_raw.convertTo(mout_rgb, CV_64FC4);
    int x = mouth_rect.origin.x;
    int y = mouth_rect.origin.y;
    int h = mouth_rect.size.height;
    int w = mouth_rect.size.width;
    cv::Rect bound(y,x-w/4, h, w);
    cv::Mat sub_mouth_rgb = mout_rgb(bound).clone();
    cv::Mat planes[4];
    cv::split(sub_mouth_rgb,planes);  // planes[2] is the red channel
    cv::Mat b_channel = planes[0].reshape(0, 1);
    cv::Mat g_channel = planes[1].reshape(0, 1);
    cv::Mat r_channel = planes[2].reshape(0, 1);
    double A_double[3][3] = {{0.299, 0.587, 0.114},{0.595716, -0.274453, -0.321263},{0.211456, -0.522591, 0.31135}};
    cv::Mat rgb_mat;
    cv::vconcat(r_channel, g_channel, rgb_mat);
    cv::vconcat(rgb_mat, b_channel, rgb_mat);
    cv::Mat A = cv::Mat(3,3,CV_64F,A_double);
    cv::Mat mou_YIQ = A*rgb_mat.clone();
    cv::Mat Q= mou_YIQ.row(2).clone();
    Q = Q.reshape(0,w)*10;
    Q.convertTo(Q, CV_8U);
    cv::cvtColor(Q, Q, CV_GRAY2RGBA);
    mouth_mask = Q;
//    cv::Mat Qth;
//    //by now using the threshold
//    cv::threshold(Q, Qth, 100, 1, cv::THRESH_BINARY);
//    cv::Mat g = cv::Mat::zeros(w, h, CV_8UC1);
//    cv::vector<cv::Mat> channels;
//    channels.push_back(g);
//    channels.push_back(g);
//    channels.push_back(Qth*200);
//    channels.push_back(Qth);
//    merge(channels, mouth_mask);
    finished_processing = true;
}

- (void)setupFaceTrackingViews
{
    faceFeatureTrackingView_ = [[UIView alloc] initWithFrame:CGRectZero];
    faceFeatureTrackingView_.layer.borderColor = [[UIColor redColor] CGColor];
    faceFeatureTrackingView_.layer.borderWidth = 3;
    faceFeatureTrackingView_.backgroundColor = [UIColor clearColor];
    faceFeatureTrackingView_.hidden = YES;
    faceFeatureTrackingView_.userInteractionEnabled = NO;
    [self.view addSubview:faceFeatureTrackingView_]; // Add as a sub-view
}

//setop the view four mouth contour
-(void)setupContourMouth
{
    contourmouth_ = [[UIImageView alloc] initWithFrame:CGRectZero];
    contourmouth_.backgroundColor = [UIColor clearColor];
    contourmouth_.hidden = YES;
    contourmouth_.userInteractionEnabled = NO;
    contourmouth_.alpha = 0.5;
    [self.view addSubview:contourmouth_];
}
// Update the face feature tracking view
- (void)updateFaceFeatureTrackingViewWithObjects:(NSArray *)objects
{
    if (!objects.count) {
        faceFeatureTrackingView_.hidden = YES;
        contourmouth_.hidden = YES;

    }
    else {
        CIFaceFeature * feature = objects[0];
        CGRect face = feature.bounds;
    
        face = CGRectApplyAffineTransform(face, portraitRotationTransform_);
        face = CGRectApplyAffineTransform(face, cameraOutputToPreviewFrameTransform_);
        faceFeatureTrackingView_.frame = face;
        faceFeatureTrackingView_.hidden = NO;
            if (feature.hasMouthPosition) {
            
            CGPoint mouth_p = feature.mouthPosition;
            CGRect mouth_r = CGRectMake(mouth_p.x-face.size.width*0.05, mouth_p.y-face.size.height*0.35, face.size.width*0.5, face.size.height*0.7);
            
            //this part if for processing matrix, first step, remove wired number
                contourmouth_.hidden = NO;
            if(10<mouth_r.size.height && mouth_r.size.height<200 && mouth_r.size.width>10&& mouth_r.size.width<200)
            {
               
   
                if (mouth_mask.rows!=0) {
 
                contourmouth_.image = [self UIImageFromCVMat:mouth_mask];
                }
                mouth_rect = mouth_r;
                mouth_r = CGRectApplyAffineTransform(mouth_r, portraitRotationTransform_);
                mouth_r = CGRectApplyAffineTransform(mouth_r, cameraOutputToPreviewFrameTransform_);
                 contourmouth_.frame = mouth_r;
               
                have_mouth = true;
            }
            else
            {
                have_mouth = false;
            }
        }
    }
}


// Calculate transformations for displaying output on the screen
- (void)calculateTransformations
{
    NSInteger outputHeight = [[detector_.captureSession.outputs[0] videoSettings][@"Height"] integerValue];
    NSInteger outputWidth = [[detector_.captureSession.outputs[0] videoSettings][@"Width"] integerValue];
    
    if (UIInterfaceOrientationIsPortrait(detector_.outputImageOrientation)) {
        // Portrait mode, swap width & height
        NSInteger temp = outputWidth;
        outputWidth = outputHeight;
        outputHeight = temp;
    }
    
    // Use self.view because self.cameraView is not resized at this point (if 3.5" device)
    CGFloat viewHeight = self.view.frame.size.height;
    CGFloat viewWidth = self.view.frame.size.width;
    
    // Calculate the scale and offset of the view vs the camera output
    // This depends on the fillmode of the GPUImageView
    CGFloat scale;
    CGAffineTransform frameTransform;
    switch (cameraView_.fillMode) {
        case kGPUImageFillModePreserveAspectRatio:
            scale = MIN(viewWidth / outputWidth, viewHeight / outputHeight);
            frameTransform = CGAffineTransformMakeScale(scale, scale);
            frameTransform = CGAffineTransformTranslate(frameTransform, -(outputWidth * scale - viewWidth)/2, -(outputHeight * scale - viewHeight)/2 );
            break;
        case kGPUImageFillModePreserveAspectRatioAndFill:
            scale = MAX(viewWidth / outputWidth, viewHeight / outputHeight);
            frameTransform = CGAffineTransformMakeScale(scale, scale);
            frameTransform = CGAffineTransformTranslate(frameTransform, -(outputWidth * scale - viewWidth)/2, -(outputHeight * scale - viewHeight)/2 );
            break;
        case kGPUImageFillModeStretch:
            frameTransform = CGAffineTransformMakeScale(viewWidth / outputWidth, viewHeight / outputHeight);
            break;
    }
    cameraOutputToPreviewFrameTransform_ = frameTransform;
    
    // In portrait mode, need to swap x & y coordinates of the returned boxes
    if (UIInterfaceOrientationIsPortrait(detector_.outputImageOrientation)) {
        // Interchange x & y
        portraitRotationTransform_ = CGAffineTransformMake(0, 1, 1, 0, 0, 0);
    }
    else {
        portraitRotationTransform_ = CGAffineTransformIdentity;
    }
    
    // AVMetaDataOutput works in texels (relative to the image size)
    // We need to transform this to pixels through simple scaling
    texelToPixelTransform_ = CGAffineTransformMakeScale(outputWidth, outputHeight);
}

-(IBAction)slidertheslider:(id)sender {
    slider.hidden = NO;
    label1.text = [NSString stringWithFormat:@"%1.1f",slider.value];
    
}
// Member functions for converting from UIImage to cvMat
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
