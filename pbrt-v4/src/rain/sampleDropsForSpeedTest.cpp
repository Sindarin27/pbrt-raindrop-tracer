//
// Created by sinda on 2024-09-12.
//
#include <pbrt/pbrt.h>
#include <cstdio>
#include <string>
#include <vector>
#include "rain.h"
#include "pbrt/util/vecmath.h"
#include "pbrt/media.h"
#include <pbrt/util/args.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

// 0 = stratified jitter with overlap
// 1 = stratified jitter with overlap and safety check
// 2 = stratified jitter without overlap
#define METHOD 2

using Allocator = pstd::pmr::polymorphic_allocator<std::byte>;
using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr, R"(usage: rain [<options>]

Options:
    --rate <mm/h>               Rain rate in mm/h
    --numSamples <samples>      Number of samples to take
    --help                      Print this help text.
)");
}

bool shouldDiscardDrop(int cellX, int cellY, int cellZ, DotMedium *medium, Float radius, Vector3f center) {
    for (int otherCellX = cellX - 1; otherCellX <= cellX + 1; otherCellX++)
        for (int otherCellY = cellY - 1; otherCellY <= cellY + 1; otherCellY++)
            for (int otherCellZ = cellZ - 1; otherCellZ <= cellZ + 1; otherCellZ++) {
                if (otherCellX == cellX && otherCellY == cellY && otherCellZ == cellZ) return false; // Arrived at self means finished
                if (!medium->cellHasDrop(otherCellX, otherCellY, otherCellZ)) continue;
                Float otherRadius = medium->cellDropRadius(otherCellX, otherCellY, otherCellZ);
                Float bothRadius = radius + otherRadius;
                Vector3f otherPos = medium->cellDropPosition(otherCellX, otherCellY, otherCellZ, otherRadius);
                if (LengthSquared(center - otherPos) < bothRadius * bothRadius) return true;
            }
    throw; // Should never reach
}

int main(int argc, char *argv[]) {
    // Convert command-line arguments to vector of strings
    std::vector<std::string> args = GetCommandLineArguments(argv);
    std::string volumeDescription;
    Point3f min = Point3f(0, 0, 0), max = Point3f(10, 10, 10), p0 = Point3f(0, 0, 0), p1 = Point3f(0.01f, 0.01f, 0.01f);
    Float rate = 50;
    int numSamples = 100000000;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        std::string tempString;

        if (ParseArg(&iter, args.end(), "rate", &rate, onError)
            || ParseArg(&iter, args.end(), "numSamles", &numSamples, onError)) {
            // success
        } else if (*iter == "--help" || *iter == "-help" || *iter == "-h") {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *iter));
            return 1;
        }
    }

    // Create medium to sample from
    Allocator alloc = Allocator();
    Transform renderFromMedium = Transform(SquareMatrix<4>::Diag(1, 1, 1, 1)); // identity matrix
    Spectrum sigma_a = Spectrum();
    Spectrum sigma_s = Spectrum();
    Float shapeParameterInv = DotMedium::CalculateDsdInvShapeParameter(rate);
    Float scaleParameter = DotMedium::CalculateDsdScaleParameterCentimeterRadius(rate);
    Bounds3f mediumBounds = Bounds3f(p0, p1);
    DotMedium *medium = alloc.new_object<DotMedium>(
            mediumBounds, renderFromMedium, sigma_a, sigma_s, 1, 0, shapeParameterInv, scaleParameter, 987654321,
            alloc);
    Point3f minInMedium = Point3f(mediumBounds.Offset(min));
    Point3f maxInMedium = Point3f(mediumBounds.Offset(max));


    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<Float> distX(minInMedium.x, maxInMedium.x);
    std::uniform_real_distribution<Float> distY(minInMedium.y, maxInMedium.y);
    std::uniform_real_distribution<Float> distZ(minInMedium.z, maxInMedium.z);

    int samplesThatHitADrop = 0;
    // Only start measuring time right before the actual method
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Loop over every grid cell
    for (int i = 0; i < numSamples; i++) {
        Float sampleX = distX(rng);
        Float sampleY = distY(rng);
        Float sampleZ = distZ(rng);
        Point3f p(sampleX, sampleY, sampleZ);
        Float sampleCellX = std::floor(p.x + Float(0.5)),
                sampleCellY = std::floor(p.y + Float(0.5)),
                sampleCellZ = std::floor(p.z + Float(0.5));
#if METHOD == 2
        Float cellX = sampleCellX;
        Float cellZ = sampleCellZ;
        Float cellY = sampleCellY;
        {
#else
        for (Float cellX = sampleCellX - 1; cellX < sampleCellX + Float(.5); cellX = std::floor(
                cellX + Float(1.1))) // NOLINT(*-flp30-c) using Float for-loop to avoid fidgeting with integers
            for (Float cellZ = sampleCellZ - 1; cellZ < sampleCellZ + Float(.5); cellZ = std::floor(
                    cellZ + Float(1.1))) // NOLINT(*-flp30-c) using Float for-loop to avoid fidgeting with integers
                for (Float cellY = sampleCellY - 1; cellY < sampleCellY + Float(.5); cellY = std::floor(cellY +
                                                                                                        Float(1.1))) { // NOLINT(*-flp30-c) using Float for-loop to avoid fidgeting with integers
#endif
                    if (medium->cellHasDrop(cellX, cellY, cellZ)) { // If this point's cell has a raindrop
//                if (cellX > 0 && cellY > 0 && cellZ > 0 && cellX < 55 && cellY < 55 && cellZ < 55) 
//                    fprintf(stderr, "%f,%f,%f\n", cellX, cellY, cellZ);

                        // Calculate position of raindrop
                        Float radius = medium->cellDropRadius(cellX, cellY, cellZ);
                        Vector3f center = medium->cellDropPosition(cellX, cellY, cellZ, radius);

                        Float diffX = center.x - p.x;
                        Float diffY = center.y - p.y;
                        Float diffZ = center.z - p.z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ < radius) {
#if METHOD == 1
                            if (shouldDiscardDrop(cellX, cellY, cellZ, medium, radius, center)) continue;                      
#endif
                            samplesThatHitADrop++;
                        };
                    }
                }

    }

    // Stop measuring time before IO stuff
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    fprintf(stderr, "%i samples hit a drop\n", samplesThatHitADrop);
    fprintf(stderr, "Took %i samples in %lli microseconds.", numSamples, duration.count());
    return 0;
}