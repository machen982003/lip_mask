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
#include <iostream>
#include "armadillo"

using namespace std;
using namespace arma;
@interface ViewController () {
    // Setup the view (this time using GPUImageView)
    GPUImageView *cameraView_;
    SMKDetectionCamera *detector_; // Detector that should be used
    UIView *faceFeatureTrackingView_; // View for showing bounding box around the face
    UIView *mouthFeatureTrackingView_;
    UIView *contourmouth_;
    CGAffineTransform cameraOutputToPreviewFrameTransform_;
    CGAffineTransform portraitRotationTransform_;
    CGAffineTransform texelToPixelTransform_;
    GPUImageRawDataOutput *rawDataOutput; //can process the gpu data
    fmat global_q;
     // save the gpu image

}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    
    // Setup GPUImageView (not we are not using UIImageView here).........
    cameraView_ = [[GPUImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
    
    // Set the face detector to be used
    detector_ = [[SMKDetectionCamera alloc] initWithSessionPreset:AVCaptureSessionPreset640x480 cameraPosition:AVCaptureDevicePositionFront];
   // [self.view addSubview:resultView_];
    rawDataOutput = [[GPUImageRawDataOutput alloc]initWithImageSize:CGSizeMake(self.view.frame.size.width, self.view.frame.size.height) resultsInBGRAFormat:true];
    [detector_ setOutputImageOrientation:UIInterfaceOrientationPortrait]; // Set to portrait
    cameraView_.fillMode = kGPUImageFillModePreserveAspectRatio;
    GPUImageMedianFilter *medianFilter = [[GPUImageMedianFilter alloc] init];
    [detector_ addTarget:rawDataOutput];
    [detector_ addTarget:medianFilter];
    [medianFilter addTarget:cameraView_];

    // Important: add as a subview
    [self.view addSubview:cameraView_];
    // Setup the face box view
    [self setupFaceTrackingViews];
    [self setupMouthTrackingViews];
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
    
    // Finally start the camera
    [detector_ startCameraCapture];
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
//setup the view for mouth tracking
- (void)setupMouthTrackingViews
{
    mouthFeatureTrackingView_ = [[UIView alloc] initWithFrame:CGRectZero];
    mouthFeatureTrackingView_.layer.borderColor = [[UIColor redColor] CGColor];
    mouthFeatureTrackingView_.layer.borderWidth = 3;
    mouthFeatureTrackingView_.backgroundColor = [UIColor clearColor];
    mouthFeatureTrackingView_.hidden = YES;
    mouthFeatureTrackingView_.userInteractionEnabled = NO;
    [self.view addSubview:mouthFeatureTrackingView_]; // Add as a sub-view
}

//setop the view four mouth contour
-(void)setupContourMouth
{
    contourmouth_ = [[UIView alloc] initWithFrame:CGRectZero];
    contourmouth_.layer.borderColor = [[UIColor yellowColor] CGColor];
    contourmouth_.layer.borderWidth = 2;
    contourmouth_.backgroundColor = [UIColor yellowColor];
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
        mouthFeatureTrackingView_.hidden = YES;
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
            mouthFeatureTrackingView_.frame = mouth_r;
            mouthFeatureTrackingView_.hidden = NO;
            //this part if for processing matrix, first step, remove wired number
            
            if(10<mouth_r.size.height && mouth_r.size.height<200 && mouth_r.size.width>10&& mouth_r.size.width<200)
            {
          //      contourmouth_.hidden = NO;
                contourmouth_.frame = mouth_r;
                __unsafe_unretained GPUImageRawDataOutput *weakOutput = rawDataOutput;
                __unsafe_unretained ViewController *weakself = self;
                __unsafe_unretained UIView *weakcontour = contourmouth_;
                [rawDataOutput setNewFrameAvailableBlock:^{
                    weakcontour.hidden = NO;
                    [weakOutput lockFramebufferForReading];
                    GLubyte *outputByte = [weakOutput rawBytesForImage];
                    fmat mou_rgb(3, int(mouth_r.size.width)*int(mouth_r.size.height), fill::zeros);
                    for(int i=0; i<mouth_r.size.height-1;i++){
                        for(int j = 0; j<mouth_r.size.width-1;j++){
                            for(int k = 0; k<3; k++){
                                mou_rgb.at(2-k,i*mouth_r.size.width+j) = outputByte[((j+int(mouth_r.origin.x))+(i+int(mouth_r.origin.y))*int(self.view.frame.size.width))*4+k];
                            }
                        }
                    }
                    fmat A = "0.299 0.587 0.114; 0.595716 -0.274453 -0.321263;0.211456 -0.622591 0.31135";
                    fmat mou_YIQ = A*mou_rgb;
                    fmat Q= mou_YIQ.row(2);
                    //  cout << mou_YIQ.n_cols<<endl;
                    Q.reshape(int(mouth_r.size.height), int(mouth_r.size.width));
                    [weakself donothing:Q];
                    [weakOutput unlockFramebufferAfterReading];
            }];
            }
        }
    }
}

-(void)donothing:(fmat) Q
{
   // cout<<Q.n_cols<<" "<<Q.n_rows<<endl;
    fmat U(size(Q),fill::ones);
    U = U*2;
    uword width = U.n_cols;
    uword height = U.n_rows;
    fmat sub_matrix(int(height*0.4),int(width*0.8),fill::zeros);
    sub_matrix = sub_matrix -2;
    U.submat(int(height*0.3), int(width*0.1), int(height*0.3)+int(height*0.4)-1, int(width*0.1)+int(width*0.8)-1) = sub_matrix;
    U = [self acwe:U image:Q];
}

-(fmat)acwe:(fmat)U image:(fmat) Q
{
    
    return U;
    int mu=1;
    int lambda1=1; int lambda2=1;
    int pc = 1;
    float pi = 3.1415926;
    float timestep = .1; int v=10; int epsilon=1; int numIter = 5;
    
    for (int k1=1; k1< numIter; k1++){
        U=[self NeumannBoundCond:U];
        fmat DrcU = pow(pow(U,2)+pow(epsilon,2),-1);
        DrcU = DrcU * (epsilon/pi);
        fmat Hu= (atan(U/epsilon)*(2/pi)+1)*0.5;
    }
    
//    K=curvature_central(u);
//    DrcU=(epsilon/pi)./(epsilon^2+u.^2);                %eq.(9), ref[2] the delta function
//    Hu=0.5*(1+(2/pi)*atan(u./epsilon));                 %eq.(8)[2] the character function how large is 'epsilon'?
//    th = .5;
//    inside_idx = find(Hu(:) < th); outside_idx = find(Hu(:) >= th);
//    c1 = mean(Img(inside_idx)); c2 = mean(Img(outside_idx));
//    data_force = -DrcU.*(mu*K - v - lambda1*(Img-c1).^2 + lambda2*(Img-c2).^2);
//    %introduce the distance regularation term:
//    P=pc*(4*del2(u) - K);               %ref[2]
//    u = u+timestep*(data_force+P);
//    end
    
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


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
