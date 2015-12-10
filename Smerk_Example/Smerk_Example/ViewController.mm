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
    rawDataOutput = [[GPUImageRawDataOutput alloc] initWithImageSize:CGSizeMake(640,480) resultsInBGRAFormat:true];
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
    // cout<<Q.n_cols<<" "<<Q.n_rows<<endl;
    finished_processing = false;
    int col_num = floor(mouth_rect.size.width);
    int row_nun = floor(mouth_rect.size.height);
    cv::Mat mou_rgb(3, col_num*row_nun,CV_8U);
    
    for(int i=0; i<row_nun;i++){
        for(int j = 0; j<col_num;j++){
            for(int k = 0; k<3; k++){
                int temp_col = j + floor(mouth_rect.origin.x);
                int temp_row = i + floor(mouth_rect.origin.y);
                mou_rgb.at<char>(2-k,i*col_num+j)= outputByte[(temp_col*int(self.view.frame.size.width)+temp_row)*4+k];
            }
        }
    }
//  //  cout<< rgb.col(0)<<"  "<<mou_rgb.col(0)<<endl;
//    double A_double[3][3] = {{0.299, 0.587, 0.114},{0.595716, -0.274453, -0.321263},{0.211456, -0.522591, 0.31135}};
//    cv::Mat A = cv::Mat(3,3,CV_64F,A_double);
//    cv::Mat mou_YIQ = A*mou_rgb.clone();
////    cout<<mou_YIQ<<endl;
//    cv::Mat Q= mou_YIQ.row(2).clone();
//    Q = Q.reshape(0,int(mouth_rect.size.height));
//    mouth_mask = Q.clone();
    cv::Mat redchannel = mou_rgb.row(0).clone();
 //   cout<<redchannel<<endl;
    cv::Mat redchannel1 = redchannel.reshape(0,row_nun);
 //   cout<<redchannel<<endl;
                                    
    mouth_mask = redchannel1.clone();
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
    contourmouth_.layer.borderColor = [[UIColor redColor] CGColor];
    contourmouth_.layer.borderWidth = 3;
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
            mouth_r = CGRectApplyAffineTransform(mouth_r, portraitRotationTransform_);
            mouth_r = CGRectApplyAffineTransform(mouth_r, cameraOutputToPreviewFrameTransform_);
            //this part if for processing matrix, first step, remove wired number
                contourmouth_.hidden = NO;
            if(10<mouth_r.size.height && mouth_r.size.height<200 && mouth_r.size.width>10&& mouth_r.size.width<200)
            {
                contourmouth_.frame = mouth_rect;
                if (mouth_mask.rows!=0) {
                cv::Mat present_mast = mouth_mask.clone();
                cv::Mat cvImage(int(mouth_mask.rows),int( mouth_mask.cols),CV_8UC3,cv::Scalar(0,0,0));
                for(int i = 0 ; i < mouth_mask.rows; i++){
                    for(int j = 0; j<mouth_mask.cols; j++){
                        cvImage.at<cv::Vec3b>(i,j)[0] =char(present_mast.at<Float64>(i, j));
                        cvImage.at<cv::Vec3b>(i,j)[1] = char(present_mast.at<Float64>(i, j));
                        cvImage.at<cv::Vec3b>(i,j)[2] = char(present_mast.at<Float64>(i, j));
                    }
                }
                
                cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
                cv::Mat display_im; cv::cvtColor(gray,display_im,CV_GRAY2BGR); // Get the display image
                
                cv::cvtColor(display_im, display_im, CV_BGR2RGBA);
 
                contourmouth_.image = [self UIImageFromCVMat:display_im];
                }
      
               mouth_rect = mouth_r;
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
