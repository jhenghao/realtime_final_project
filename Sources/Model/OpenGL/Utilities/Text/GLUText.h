/*
     File: GLUText.h
 Abstract: 
 Utility methods for generating OpenGL texture from a string.
 
  Version: 3.3
 
 Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple
 Inc. ("Apple") in consideration of your agreement to the following
 terms, and your use, installation, modification or redistribution of
 this Apple software constitutes acceptance of these terms.  If you do
 not agree with these terms, please do not use, install, modify or
 redistribute this Apple software.
 
 In consideration of your agreement to abide by the following terms, and
 subject to these terms, Apple grants you a personal, non-exclusive
 license, under Apple's copyrights in this original Apple software (the
 "Apple Software"), to use, reproduce, modify and redistribute the Apple
 Software, with or without modifications, in source and/or binary forms;
 provided that if you redistribute the Apple Software in its entirety and
 without modifications, you must retain this notice and the following
 text and disclaimers in all such redistributions of the Apple Software.
 Neither the name, trademarks, service marks or logos of Apple Inc. may
 be used to endorse or promote products derived from the Apple Software
 without specific prior written permission from Apple.  Except as
 expressly stated in this notice, no other rights or licenses, express or
 implied, are granted by Apple herein, including but not limited to any
 patent rights that may be infringed by your derivative works or by other
 works in which the Apple Software may be incorporated.
 
 The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
 MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
 THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
 OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
 
 IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
 MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
 AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
 STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
 
 Copyright (C) 2014 Apple Inc. All Rights Reserved.
 
 */

#ifndef _OPENGL_UTILITY_TEXT_H_
#define _OPENGL_UTILITY_TEXT_H_

#import <string>

#import <Cocoa/Cocoa.h>
#import <OpenGL/OpenGL.h>

#import "CTFrame.h"
#import "GLcontainers.h"

#ifdef __cplusplus

namespace GLU
{
    class Text
    {
    public:
        // Create a texture with bounds derived from the text size.
        Text(const GLstring& rText,
             const GLstring& rFont,
             const GLfloat& nFontSize,
             const CGPoint& rOrigin,
             const CTTextAlignment& nTextAlign = kCTTextAlignmentCenter);
        
        // Create a texture with bounds derived from the input width and height.
        Text(const GLstring& rText,
             const GLstring& rFont,
             const GLfloat& nFontSize,
             const GLsizei& nWidth,
             const GLsizei& nHeight,
             const CTTextAlignment& nTextAlign = kCTTextAlignmentCenter);
        
        // Create a texture with bounds derived from the text size using
        // helvetica bold or helvetica bold oblique font.
        Text(const GLstring& rText,
             const CGFloat& nFontSize,
             const bool& bIsItalic,
             const CGPoint& rOrigin,
             const CTTextAlignment& nTextAlign = kCTTextAlignmentCenter);
        
        // Create a texture with bounds derived from input width and height,
        // and using helvetica bold or helvetica bold oblique font.
        Text(const GLstring& rText,
             const CGFloat& nFontSize,
             const bool& bIsItalic,
             const GLsizei& nWidth,
             const GLsizei& nHeight,
             const CTTextAlignment& nTextAlign = kCTTextAlignmentCenter);
        
        virtual ~Text();
        
        const GLuint&  texture() const;
        const CGRect&  bounds()  const;
        const CFRange& range()   const;
        
    private:
        CGContextRef create(const GLsizei& nWidth,
                            const GLsizei& nHeight);
        
        CGContextRef create(const CGSize& rSize);
        
        GLuint create(CGContextRef pContext);
        
        GLuint create(const GLstring& rText,
                      const GLstring& rFont,
                      const GLfloat& nFontSize,
                      const CGPoint& rOrigin,
                      const CTTextAlignment& nTextAlign);
        
        GLuint create(const GLstring& rText,
                      const GLstring& rFont,
                      const GLfloat& nFontSize,
                      const GLsizei& nWidth,
                      const GLsizei& nHeight,
                      const CTTextAlignment& nTextAlign);
        
    private:
        GLuint   mnTexture;
        CGRect   m_Bounds;
        CFRange  m_Range;
    }; // Text
} // GLU

#endif

#endif


