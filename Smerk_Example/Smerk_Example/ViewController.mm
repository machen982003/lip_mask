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
#include "armadillo"
#include <opencv2/opencv.hpp>
#include <math.h>
#endif

#define PI 3.1415926

using namespace std;
using namespace arma;
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
    fmat mouth_mask;
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
    upb =1;
    btb = 400;
    [rawDataOutput setNewFrameAvailableBlock:^{
        upb = 200;
        btb = 300;
        [weakself->rawDataOutput lockFramebufferForReading];
     
        GLubyte *outputByte = weakself->rawDataOutput.rawBytesForImage;
        if (have_mouth == true && finished_processing == true) {
            upb =1;
            btb = 400;
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
   // cube test(reinterpret_cast<char*>(outputByte),10 ,10,4);
//    fcube test1 = test.subcube(int(mouth_rect.origin.x), int(mouth_rect.origin.y), 0, int(mouth_rect.origin.x+mouth_rect.size.width-1), int(mouth_rect.origin.y+mouth_rect.size.height-1), 2);
////    
//    fmat b = test1.slice(0);
//    b.reshape(1,int(mouth_rect.size.width)*int(mouth_rect.size.height));
//    fmat g = test1.slice(1);
//    g.reshape(1,int(mouth_rect.size.width)*int(mouth_rect.size.height));
//    fmat r = test1.slice(2);
//    r.reshape(1,int(mouth_rect.size.width)*int(mouth_rect.size.height));
//    fmat rgb(3,int(mouth_rect.size.width)*int(mouth_rect.size.height), fill::zeros);
//    rgb.row(0) = r;
//    rgb.row(1) = g;
//    rgb.row(2) = b;
    
    fmat mou_rgb(3, int(mouth_rect.size.width)*int(mouth_rect.size.height), fill::zeros);
    
    for(int i=0; i<mouth_rect.size.height-1;i++){
        for(int j = 0; j<mouth_rect.size.width-1;j++){
            for(int k = 0; k<3; k++){
                mou_rgb.at(2-k,i*mouth_rect.size.width+j) = outputByte[((j+int(mouth_rect.origin.x))+(i+int(mouth_rect.origin.y))*480)*4+k];
            }
        }
    }
  //  cout<< rgb.col(0)<<"  "<<mou_rgb.col(0)<<endl;
    fmat A = "0.299 0.587 0.114; 0.595716 -0.274453 -0.321263;0.211456 -0.622591 0.31135";
    fmat mou_YIQ = A*mou_rgb;
    fmat Q= mou_YIQ.row(2);
    //  cout << mou_YIQ.n_cols<<endl;
    Q.reshape(int(mouth_rect.size.height), int(mouth_rect.size.width));
    fmat U(size(Q),fill::ones);
    U = U*2;
    uword width = U.n_cols;
    uword height = U.n_rows;
    fmat sub_matrix(int(height*0.4),int(width*0.8),fill::zeros);
    sub_matrix = sub_matrix -2;
    U.submat(int(height*0.3), int(width*0.1), int(height*0.3)+int(height*0.4)-1, int(width*0.1)+int(width*0.8)-1) = sub_matrix;

    [self acwe:U image:Q];
//    cout<<U<<endl;
//    cv::Mat cvfromarm(U.n_cols, U.n_rows, CV_32FC1, U.memptr());
//    cv::Mat cvfinal(cvfromarm.t());
//    contourmouth_.image = [self UIImageFromCVMat:cvfinal];
    finished_processing = true;
}

//get the contour of mouth
-(void)acwe:(fmat)Uinput image:(fmat) Qinput
{
    
    fmat U = Uinput;
    fmat Q = Qinput;
    int mu=1;
    int lambda1=1; int lambda2=1;
    float timestep = .1; int v=10; int epsilon=1; int numIter = 5;
    
    for (int k1=1; k1< numIter; k1++){
        U=[self NeumannBoundCond:U];
        fmat K=[self curvature_central:U];
        fmat DrcU = pow(pow(U,2)+pow(epsilon,2),-1);
        DrcU = DrcU * (epsilon/PI);
        fmat Hu= (atan(U/epsilon)*(2/PI)+1)*0.5;
        float th = 0.5;
        uvec inside_idx = find(Hu<th);
        uvec outside_idx = find(Hu>=th);
        float c1 = mean(Q.elem(inside_idx));
        float c2 = mean(Q.elem(outside_idx));
        fmat data_force = -DrcU%(mu*K - v- lambda1*pow((Q-c1),2) +lambda2*pow((Q-c2), 2));
        //    P=pc*(4*del2(u) - K);               %ref[2]
        U = U+timestep*data_force;
    }
    U.elem((find(U>0))).zeros();
    U.elem((find(U<0))).ones();
    mouth_mask = U;
    
}

-(fmat)NeumannBoundCond:(fmat) f
{
    uword nrow = f.n_rows;
    uword ncol = f.n_cols;
    fmat g = f;
    //change 4 corner points
    g.at(0, 0) = g.at(2, 2);
    g.at(0, ncol-1) = g.at(2,ncol-3);
    g.at(nrow-1,0) = g.at(nrow-3,2);
    g.at(nrow-1,ncol-1) = g.at(nrow-3,ncol-3);
    //change top and bottom edge
    g.submat(0, 1, 0, ncol-2) = g.submat(2, 1, 2, ncol-2);
    g.submat(nrow-1, 1, nrow-1, ncol-2) = g.submat(nrow-3, 1, nrow-3, ncol-2);
    //change left and right edge
    g.submat(1, 0, nrow-2, 0) = g.submat(1, 2, nrow-2, 2);
    g.submat(1, ncol-1, nrow-2, ncol-1) = g.submat(1, ncol-3, nrow-2, ncol-3);
    return g;
}

-(fmat)curvature_central:(fmat) u
{
    fmat k;
    fmat gx = [self gradient:u andIfXdirection:true];
    fmat gy = [self gradient:u andIfXdirection:false];
    fmat normDu = sqrt(pow(gx,2)+pow(gy,2)+0.00000001);
    fmat Nx = gx/normDu;
    fmat Ny = gy/normDu;
    fmat nxx = [self gradient:Nx andIfXdirection:true];
    fmat nyy = [self gradient:Ny andIfXdirection:false];
    k = nxx+nyy;
    return k;
    
}

-(fmat)gradient:(fmat) input andIfXdirection:(BOOL) direction
{
    fmat g(size(input), fill::zeros);
    //get the x direction gradient
    if(direction){
        uword n = input.n_rows;
        if (n>1) {
            g.row(0) = input.row(1) -input.row(0);
            g.row(n-1) = input.row(n-1) - input.row(n-2);
        }
        if (n>2){
            g.rows(1, n-2) = (input.rows(2,n-1) - input.rows(0,n-3))/2;
            
        }
    }
    //get the y direction gradient
    else{
        uword n = input.n_cols;
        if(n>1){
            g.col(0) = input.col(1) - input.col(0);
            g.col(n-1) = input.col(n-1) - input.col(n-2);
            
        }
        if (n>2) {
            g.cols(1, n-2) = (input.cols(2, n-1) - input.cols(0, n-3))/2;
        }
        
    }
    return g;
    
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
         
            if(10<mouth_r.size.height && mouth_r.size.height<200 && mouth_r.size.width>10&& mouth_r.size.width<200)
            {
                contourmouth_.frame = mouth_rect;
                cv::Mat cvImage(int(mouth_mask.n_cols),int( mouth_mask.n_rows),CV_8UC3,cv::Scalar(100,0,0));
                for(int i = 0 ; i < mouth_mask.n_rows; i++){
                    for(int j = 0; j<mouth_mask.n_cols; j++){
                        cvImage.at<cv::Vec3b>(i,j)[0] = mouth_mask.at(i, j);
                        cvImage.at<cv::Vec3b>(i,j)[1] = mouth_mask.at(i, j);
                        cvImage.at<cv::Vec3b>(i,j)[2] = mouth_mask.at(i, j);
                    }
                }
      //          cvImage.rowRange(upb, btb) = 255;
                
               cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
                cv::Mat display_im; cv::cvtColor(gray,display_im,CV_GRAY2BGR); // Get the display image
                
                cv::cvtColor(display_im, display_im, CV_BGR2RGBA);
                contourmouth_.image = [self UIImageFromCVMat:display_im];
                contourmouth_.hidden = NO;
                
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
