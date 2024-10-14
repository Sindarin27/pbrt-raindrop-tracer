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

using Allocator = pstd::pmr::polymorphic_allocator<std::byte>;
using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr, R"(usage: rain [<options>]

Options:
    --rate <mm/h>               Rain rate in mm/h
    --method <0/1/2>          Method to use for generating drops. 0=stratified jitter with cell overlap, 1=0 with safety checks, 2=stratified jitter without overlap
    --outfile <filename.pbrt>   Output file containing object definitions
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

    Point3f min = Point3f(0,0,0), max = Point3f(10,10,10), p0 = Point3f(0, 0, 0), p1 = Point3f(0.01f, 0.01f, 0.01f);
    Float rate = 50;
    std::string outFilename;
    // 0 = stratified jitter with overlap
    // 1 = stratified jitter with overlap and safety check
    // 2 = stratified jitter without overlap
    int method = 0;
    
    for (auto iter = args.begin(); iter != args.end(); ++iter) {

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        std::string tempString;

        if (ParseArg(&iter, args.end(), "rate", &rate, onError)
                   || ParseArg(&iter, args.end(), "outfile", &outFilename, onError)
                   || ParseArg(&iter, args.end(), "method", &method, onError)) {
            // success
        } else if (*iter == "--help" || *iter == "-help" || *iter == "-h") {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *iter));
            return 1;
        }
    }
    if (outFilename.empty()) {
        usage("Didn't find a value for --outfile");
        return 1;
    }

    fprintf(stderr, "rain: %s\n\n Raining at a rate of %f to file %s\n", "hello world!", rate, outFilename.c_str());
    // Ensure file path exists
    std::filesystem::path filePath = outFilename;
    std::filesystem::path fileDir = filePath.parent_path();
    if (!std::filesystem::is_directory(fileDir) || !std::filesystem::exists(fileDir)) {
        fprintf(stderr, "Creating directory %ls\n", fileDir.c_str());
        std::filesystem::create_directory(fileDir);
    }
    // Open file
    std::ofstream out;
    out.open(outFilename);
    if (!out || out.fail()) {
        fprintf(stderr, "Failed to open file %s\n", outFilename.c_str());
        return 1;
    }

    // Set settings for stream
    out << std::fixed // Set fixed floating-point
        << std::setprecision(
                std::numeric_limits<Float>::max_digits10); // Set precision to number of decimal digits required

    // Create medium to sample from
    Allocator alloc = Allocator();
    Transform renderFromMedium = Transform(SquareMatrix<4>::Diag(1, 1, 1, 1)); // identity matrix
    Spectrum sigma_a = Spectrum();
    Spectrum sigma_s = Spectrum();
    Float shapeParameterInv = DotMedium::CalculateDsdInvShapeParameter(rate);
    Float scaleParameter = DotMedium::CalculateDsdScaleParameterCentimeterRadius(rate);
    Bounds3f mediumBounds = Bounds3f(p0, p1);
    DotMedium *medium = alloc.new_object<DotMedium>(
            mediumBounds, renderFromMedium, sigma_a, sigma_s, 1, 0, shapeParameterInv, scaleParameter, 987654321, alloc);
    Point3f minInMedium = Point3f(mediumBounds.Offset(min));
    Point3f maxInMedium = Point3f(mediumBounds.Offset(max));
    Float maxDimension = mediumBounds.Diagonal()[mediumBounds.MaxDimension()];
    int minX = std::floor(minInMedium.x);
    int minY = std::floor(minInMedium.y);
    int minZ = std::floor(minInMedium.z);
    int maxX = std::ceil(maxInMedium.x);
    int maxY = std::ceil(maxInMedium.y);
    int maxZ = std::ceil(maxInMedium.z);
    int dropCount = 0, dropCountDiscarded = 0;
    
    // Only start measuring time right before the actual method
    auto start_time = std::chrono::high_resolution_clock::now();

    // Loop over every grid cell
    for (int x = minX; x <= maxX; x++) {
            Float cellX = std::floor(Float(x) + Float(0.5));
            for (int y = minY; y <= maxY; y++) {
                Float cellY = std::floor(Float(y) + Float(0.5));
                for (int z = minZ; z <= maxZ; z++) {
                    Float cellZ = std::floor(Float(z) + Float(0.5));

                    // Check if this cell should contain a drop
                    if (!medium->cellHasDrop(cellX, cellY, cellZ)) continue;
                    //fprintf(stderr, "Cell has drop at %f,%f,%f\n", cellX, cellY, cellZ);
                    // Calculate stuff about the drop
                    Float radius = medium->cellDropRadius(cellX, cellY, cellZ);
                    
                    Vector3f center;
                    // Use different formula for stratified jitter without overlap
                    if (method == 2) {
                        Float maxShift = Float(0.5) - radius;
                        Float centerX = cellX + maxShift * HashFloatWithNeg(cellX, cellY, cellZ, 2);
                        Float centerY = cellY + maxShift * HashFloatWithNeg(cellX, cellY, cellZ, 3);
                        Float centerZ = cellZ + maxShift * HashFloatWithNeg(cellX, cellY, cellZ, 4);
                        center = {centerX, centerY, centerZ};
                    }
                    else center = medium->cellDropPosition(cellX, cellY, cellZ, radius);
                    
                    // if safe stratified jitter with overlap, check neighbours
                        
                    if (method == 1 && shouldDiscardDrop(cellX, cellY, cellZ, medium, radius, center)) {
                        dropCountDiscarded++;
                        continue;
                    }
                    
                    dropCount++;
                    // Write the drop to file
                    Vector3f start = mediumBounds.OffsetReverse(Point3f(center.x, center.y, center.z));
                    out << start.x << ";" << start.y << ";" << start.z << ";" << radius << " \n";
                }
            }
        }
    // Stop measuring time before IO stuff
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);

    // End file
    out.close();

    Vector3f span = maxInMedium - minInMedium;
    Float volume = span.x * span.y * span.z;
    fprintf(stderr, "Wrote %i drops to a file in a volume of %f cm^3, for %f drops per cm^3, discarded %i drops. Took %lli microseconds.",
            dropCount, volume, Float(dropCount) / volume, dropCountDiscarded, duration.count());
    return 0;
}