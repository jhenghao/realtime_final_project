/*
     File: NBodySimulationRandom.mm
 Abstract: 
 Functor for generating random data sets for the cpu or gpu bound simulator.
 
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

#import <fstream>
#import <iostream>

#import "GLMSizes.h"

#import "NBodySimulationRandom.h"

#pragma mark -
#pragma mark Private - Namespace

using namespace NBody::Simulation;

#pragma mark -
#pragma mark Private - Constants

static const GLchar *kGalaxyDataFiles[] =
{
    "bodies_16k.dat",
    "bodies_24k.dat",
    "bodies_32k.dat",
    "bodies_64k.dat",
    "bodies_80k.dat",
};

static const GLfloat kBodyCountScale = 1.0f / 16384.0f;
static const GLfloat kRandMax        = GLfloat(RAND_MAX);

#pragma mark -
#pragma mark Private - Utilities

simd::float3 Data::Random::create(const bool& bIsClamped)
{
    simd::float3 p = 0.0f;
    
    if(bIsClamped)
    {
        p.x = GLfloat(std::rand()) / kRandMax * 2.0f - 1.0f;
        p.y = GLfloat(std::rand()) / kRandMax * 2.0f - 1.0f;
        p.z = GLfloat(std::rand()) / kRandMax * 2.0f - 1.0f;
    } // if
    else
    {
        p.x = GLfloat(std::rand()) / kRandMax;
        p.y = GLfloat(std::rand()) / kRandMax;
        p.z = GLfloat(std::rand()) / kRandMax;
    } // else
    
    return p;
} // Data::Random::create

void Data::Random::acquire(GLfloat* pPosition,
                           GLfloat* pVelocity)
{
    const GLfloat fcount   = GLfloat(mnBodies);
    const GLfloat fbcscale = fcount / 1024.0f;
    const GLfloat ftcscale = 16384.0f / fcount;
    const GLfloat fvcscale = kBodyCountScale * fcount;
    
    switch(mnConfig)
    {
        default:
        case NBody::eConfigRandom:
        {
            GLfloat scale  = m_Scale[0] * std::max(1.0f, fbcscale);
            GLfloat vscale = m_Scale[1] * scale;
            
            GLint p = 0;
            GLint v = 0;
            GLint i = 0;
            
            simd::float3 point    = 0.0f;
            simd::float3 velocity = 0.0f;
            
            GLfloat scalar = 0.0f;
            
            while(i < mnBodies)
            {
                point  = Data::Random::create(true);
                scalar = simd::length_squared(point);
                
                if(scalar > 1)
                {
                    continue;
                } // if
                
                velocity = Data::Random::create(true);
                scalar   = simd::length_squared(velocity);
                
                if(scalar > 1)
                {
                    continue;
                } // if
                
                point    *= scale;
                velocity *= vscale;
                
                pPosition[p++] = point.x;
                pPosition[p++] = point.y;
                pPosition[p++] = point.z;
                pPosition[p++] = 1.0f; // mass
                
                pVelocity[v++] = velocity.x;
                pVelocity[v++] = velocity.y;
                pVelocity[v++] = velocity.z;
                pVelocity[v++] = 1.0f; // inverse mass
                
                i++;
            } // while
        } // NBody::eConfigRandom
            
            break;
            
        case NBody::eConfigShell:
        {
            GLfloat scale  = m_Scale[0];
            GLfloat vscale = scale * m_Scale[1];
            GLfloat inner  = 2.5f * scale;
            GLfloat outer  = 4.0f * scale;
            
            GLint p = 0;
            GLint v = 0;
            GLint i = 0;
            
            GLfloat dot = 0.0f;
            GLfloat len = 0.0f;
            
            simd::float3 point    = 0.0f;
            simd::float3 position = 0.0f;
            simd::float3 velocity = 0.0f;
            simd::float3 axis     = 0.0f;
            
            while(i < mnBodies)
            {
                point = Data::Random::create(true);
                len   = simd::length(point);
                point = simd::normalize(point);
                
                if(len > 1)
                {
                    continue;
                } // if
                
                position = Data::Random::create(false);
                position *= (outer - inner);
                position += inner;
                position *= point;
                
                pPosition[p++] = position.x;
                pPosition[p++] = position.y;
                pPosition[p++] = position.z;
                pPosition[p++] = ftcscale;
                
                axis = {0.0f, 0.0f, 1.0f};
                axis = simd::normalize(axis);
                dot  = simd::dot(point, axis);
                
                if((1.0f - dot) < 1e-6)
                {
                    axis.x = point.y;
                    axis.y = point.x;
                    
                    axis = simd::normalize(axis);
                } // if
                
                velocity  = simd::cross(position, axis);
                velocity *= vscale;
                
                pVelocity[v++] = velocity.x;
                pVelocity[v++] = velocity.y;
                pVelocity[v++] = velocity.z;
                pVelocity[v++] = fvcscale;
                
                i++;
            } // while
        } // NBody::eConfigShell
            
            break;
            
            // Galaxy collision
        case NBody::eConfigMWM31:
        {
            GLfloat scale  = m_Scale[0];
            GLfloat vscale = scale * m_Scale[1];
            GLfloat mscale = scale * scale * scale;
            
            GLint p = 0;
            GLint v = 0;
            
            std::ifstream *infile;
            
            switch (mnBodies)
            {
                case 16384:
                    infile = new std::ifstream(kGalaxyDataFiles[0]);
                    break;
                case 24576:
                    infile = new std::ifstream(kGalaxyDataFiles[1]);
                    break;
                case 32768:
                    infile = new std::ifstream(kGalaxyDataFiles[2]);
                    break;
                case 65536:
                    infile = new std::ifstream(kGalaxyDataFiles[3]);
                    break;
                case 81920:
                    infile = new std::ifstream(kGalaxyDataFiles[4]);
                    break;
                default:
                    std::cerr
                    << ">> ERROR: Number of bodies must be one of 16384, 24576, 32768, 65536, 81920, 131072 or 1048576!"
                    << std::endl;
                    
                    exit(1);
                    
                    break;
            } // switch
            
            GLint numPoints = 0;
            
            GLfloat bMass = 0.0f;
            GLfloat bIDf  = 0.0f;
            
            GLfloat value[6];
            
            simd::float3 position = 0.0f;
            simd::float3 velocity = 0.0f;
            
            if(!infile->fail())
            {
                while(!(infile->eof()) && (numPoints < mnBodies))
                {
                    numPoints++;
                    
                    *infile >> bMass
                    >> value[0]
                    >> value[1]
                    >> value[2]
                    >> value[3]
                    >> value[4]
                    >> value[5]
                    >> bIDf;
                    
                    position = {value[0], value[1], value[2]};
                    velocity = {value[3], value[4], value[5]};
                    
                    bMass *= mscale;
                    
                    position *= scale;
                    
                    pPosition[p++] = position.x;
                    pPosition[p++] = position.y;
                    pPosition[p++] = position.z;
                    pPosition[p++] = bMass;
                    
                    velocity *= vscale;
                    
                    pVelocity[v++] = velocity.x;
                    pVelocity[v++] = velocity.y;
                    pVelocity[v++] = velocity.z;
                    pVelocity[v++] = 1.0f / bMass;
                } // while
            } // if
            
            delete infile;
        } // NBody::eConfigMWM31
            
            break;
            
        case NBody::eConfigExpand:
        {
            GLfloat scale = m_Scale[0] * std::max(1.0f, mnBodies / (1024.f));
            GLfloat vscale = scale * m_Scale[1];
            GLfloat lenSqr = 0.0f;
            
            simd::float3 point    = 0.0f;
            simd::float3 position = 0.0f;
            simd::float3 velocity = 0.0f;
            
            GLint i;
            
            GLint p = 0;
            GLint v = 0;
            
            for(i = 0; i < mnBodies;)
            {
                point  = Data::Random::create(true);
                lenSqr = simd::length_squared(point);
                
                if(lenSqr > 1.0f)
                {
                    continue;
                } // if
                
                position *= scale;
                
                pPosition[p++] = position.x;
                pPosition[p++] = position.y;
                pPosition[p++] = position.z;
                pPosition[p++] = 1.0f;
                
                velocity *= vscale;
                
                pVelocity[v++] = velocity.x;
                pVelocity[v++] = velocity.y;
                pVelocity[v++] = velocity.z;
                pVelocity[v++] = 1.0f;
                
                i++;
            } // for
        } // NBody::eConfigExpand
            
            break;
    } // switch
} // acquire

void Data::Random::acquire(Split *pSplit,
                           Packed *pPacked)
{
    const GLfloat fcount = GLfloat(mnBodies);
    const GLfloat fbcscale = fcount / 1024.0f;
    
    GLfloat *pMass = pPacked->mass();
    
    GLfloat *pPositionX = pSplit->position(Data::eCoordinateX);
    GLfloat *pPositionY = pSplit->position(Data::eCoordinateY);
    GLfloat *pPositionZ = pSplit->position(Data::eCoordinateZ);
    
    GLfloat *pVelocityX = pSplit->velocity(Data::eCoordinateX);
    GLfloat *pVelocityY = pSplit->velocity(Data::eCoordinateY);
    GLfloat *pVelocityZ = pSplit->velocity(Data::eCoordinateZ);
    
    switch(mnConfig)
    {
        default:
        case NBody::eConfigRandom:
        {
            GLfloat scale  = m_Scale[0] * std::max(1.0f, fbcscale);
            GLfloat vscale = m_Scale[1] * scale;
            GLfloat lenSqr = 0.0f;
            
            GLint p = 0;
            GLint v = 0;
            GLint i = 0;
            
            simd::float3 point    = 0.0f;
            simd::float3 velocity = 0.0f;
            
            while(i < mnBodies)
            {
                point  = Data::Random::create(true);
                lenSqr = simd::length_squared(point);
                
                if(lenSqr > 1.0f)
                {
                    continue;
                } // if
                
                velocity = Data::Random::create(true);
                lenSqr   = simd::length_squared(velocity);
                
                if(lenSqr > 1.0f)
                {
                    continue;
                } // if
                
                point *= scale;
                
                pPositionX[p] = point.x;
                pPositionY[p] = point.y;
                pPositionZ[p] = point.z;
                
                pMass[p] = 1.0f; // mass
                
                velocity *= vscale;
                
                pVelocityX[v] = velocity.x;
                pVelocityY[v] = velocity.y;
                pVelocityZ[v] = velocity.z;
                
                p++;
                v++;
                i++;
            } // while
        } // NBody::eConfigRandom
            
            break;
            
        case NBody::eConfigShell:
        {
            GLfloat scale  = m_Scale[0];
            GLfloat vscale = scale * m_Scale[1];
            GLfloat inner  = 2.5f * scale;
            GLfloat outer  = 4.0f * scale;
            GLfloat len    = 0.0f;
            GLfloat dot    = 0.0f;
            
            GLint p = 0;
            GLint v = 0;
            GLint i = 0;
            
            simd::float3 point    = 0.0f;
            simd::float3 position = 0.0f;
            simd::float3 velocity = 0.0f;
            simd::float3 axis     = 0.0f;
            
            while(i < mnBodies)
            {
                point = Data::Random::create(true);
                len   = simd::length(point);
                point = simd::normalize(point);
                
                if(len > 1)
                {
                    continue;
                } // if
                
                position = Data::Random::create(false);
                position *= (outer - inner);
                position += inner;
                position *= point;
                
                pPositionX[p] = position.x;
                pPositionY[p] = position.y;
                pPositionZ[p] = position.z;
                
                pMass[p] = 1.0f;
                
                axis = {0.0f, 0.0f, 1.0f};
                axis = simd::normalize(axis);
                dot  = simd::dot(point, axis);
                
                if((1.0f - dot) < 1.0e-6)
                {
                    axis.x = point.y;
                    axis.y = point.x;
                    
                    axis = simd::normalize(axis);
                } // if
                
                velocity = {pPositionX[i], pPositionY[i], pPositionZ[i]};
                velocity = simd::cross(velocity, axis);
                velocity *= vscale;
                
                pVelocityX[v] = velocity.x;
                pVelocityY[v] = velocity.y;
                pVelocityZ[v] = velocity.z;
                
                p++;
                v++;
                i++;
            } // while
        } // NBody::eConfigShell
            
            break;
            
            // Galaxy collision
        case NBody::eConfigMWM31:
        {
            GLfloat scale = m_Scale[0];
            GLfloat vscale = scale * m_Scale[1];
            GLfloat mscale = scale * scale * scale;
            
            std::ifstream *infile;
            
            switch (mnBodies)
            {
                case 16384:
                    infile = new std::ifstream(kGalaxyDataFiles[0]);
                    break;
                case 24576:
                    infile = new std::ifstream(kGalaxyDataFiles[1]);
                    break;
                case 32768:
                    infile = new std::ifstream(kGalaxyDataFiles[2]);
                    break;
                case 65536:
                    infile = new std::ifstream(kGalaxyDataFiles[3]);
                    break;
                case 81920:
                    infile = new std::ifstream(kGalaxyDataFiles[4]);
                    break;
                default:
                    std::cerr
                    << ">> ERROR: Number of bodies must be one of 16384, 24576, 32768, 65536 or 81920!"
                    << std::endl;
                    
                    exit(1);
                    
                    break;
            } // switch
            
            GLint numPoints = 0;
            GLint p = 0;
            
            GLfloat value[6];
            
            simd::float3 position = 0.0f;
            simd::float3 velocity = 0.0f;
            
            GLfloat bMass = 0.0f;
            GLfloat bIDf  = 0.0f;
            
            if(!infile->fail())
            {
                while(!(infile->eof()) && (numPoints < mnBodies))
                {
                    numPoints++;
                    
                    *infile >> bMass
                    >> value[0]
                    >> value[1]
                    >> value[2]
                    >> value[3]
                    >> value[4]
                    >> value[5]
                    >> bIDf;
                    
                    position = {value[0], value[1], value[2]};
                    velocity = {value[3], value[4], value[5]};
                    
                    bMass *= mscale;
                    
                    position *= scale;
                    
                    pPositionX[p] = position.x;
                    pPositionY[p] = position.y;
                    pPositionZ[p] = position.z;
                    
                    pMass[p] = bMass;
                    
                    velocity *= vscale;
                    
                    pVelocityX[p] = velocity.x;
                    pVelocityY[p] = velocity.y;
                    pVelocityZ[p] = velocity.z;
                    
                    p++;
                } // while
            } // if
            
            delete infile;
        } // NBody::eConfigMWM31
            
            break;
            
        case NBody::eConfigExpand:
        {
            GLfloat scale  = m_Scale[0] * std::max(1.0f, fbcscale);
            GLfloat vscale = scale * m_Scale[1];
            GLfloat lenSqr = 0.0f;
            
            GLint p = 0;
            GLint v = 0;
            
            GLint i;
            
            simd::float3 point    = 0.0f;
            simd::float3 velocity = 0.0f;
            
            for(i = 0; i < mnBodies;)
            {
                point  = Data::Random::create(true);
                lenSqr = simd::length_squared(point);
                
                if(lenSqr > 1.0f)
                {
                    continue;
                } // if
                
                point *= scale;
                
                pPositionX[p] = point.x;
                pPositionY[p] = point.y;
                pPositionZ[p] = point.z;
                
                pMass[p] = 1.0f; // pMass
                
                velocity *= vscale;
                
                pVelocityX[v] = velocity.x;
                pVelocityY[v] = velocity.y;
                pVelocityZ[v] = velocity.z;
                
                p++;
                v++;
                i++;
            } // for
        } // NBody::eConfigExpand
            
            break;
    } // switch
} // acquire

void Data::Random::copy(Split *pSplit,
                        Packed *pPacked)
{
    GLfloat *pMass      = pPacked->mass();
    GLfloat *pPosition  = pPacked->position();
    GLfloat *pPositionX = pSplit->position(Data::eCoordinateX);
    GLfloat *pPositionY = pSplit->position(Data::eCoordinateY);
    GLfloat *pPositionZ = pSplit->position(Data::eCoordinateZ);
    
    GLuint i;
    GLuint j;
    
    for(i = 0; i < mnBodies; ++i)
    {
        j = 4 * i;
        
        pPosition[j]   = pPositionX[i];
        pPosition[j+1] = pPositionY[i];
        pPosition[j+2] = pPositionZ[i];
        pPosition[j+3] = pMass[i];
    } // for
} // copy

#pragma mark -
#pragma mark Public - Constructor

Data::Random::Random(const size_t& nBodies,
                     const Params& rParams)
{
    mnBodies   = nBodies;
    mnConfig   = rParams.mnConfig;
    m_Scale[0] = rParams.mnClusterScale;
    m_Scale[1] = rParams.mnVelocityScale;
} // Constructor

#pragma mark -
#pragma mark Public - Destructor

Data::Random::~Random()
{
    mnBodies   = 0;
    mnConfig   = eConfigRandom;
    m_Scale[0] = 0.0f;
    m_Scale[1] = 0.0f;
} // Destructor

#pragma mark -
#pragma mark Public - Accessor

void Data::Random::setParam(const Params& rParams)
{
    mnConfig   = rParams.mnConfig;
    m_Scale[0] = rParams.mnClusterScale;
    m_Scale[1] = rParams.mnVelocityScale;
} // setParam

#pragma mark -
#pragma mark Public - Operators

bool Data::Random::operator()(Split *pSplit,
                              Packed *pPacked)
{
    bool bSuccess = (pSplit != nullptr) && (pPacked != nullptr);
    
    if(bSuccess)
    {
        acquire(pSplit, pPacked);
        copy(pSplit, pPacked);
    } // if
    
    return bSuccess;
} // operator()

bool Data::Random::operator()(GLfloat *pPosition,
                              GLfloat *pVelocity)
{
    bool bSuccess = (pPosition != nullptr) && (pVelocity != nullptr);
    
    if(bSuccess)
    {
        acquire(pPosition, pVelocity);
    } // if
    
    return bSuccess;
} // operator()
