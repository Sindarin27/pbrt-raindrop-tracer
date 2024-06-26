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

using Allocator = pstd::pmr::polymorphic_allocator<std::byte>;
using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr, R"(usage: rain [<options>]

Options:
    --min <x,y,z>               Lower bound of the cube to fill with drops
    --max <x,y,z>               Upper bound of the cube to fill with drops
    --p0 <x,y,z>                Point 0 of the rain volume. Forms a 1x1x1 cm cube with p1
    --p1 <x,y,z>                Point 1 of the rain volume. Forms a 1x1x1 cm cube with p0
    --rate <mm/h>               Rain rate in mm/h
    --outfile <filename.pbrt>   Output file containing object definitions
    --help                      Print this help text.
)");
}

int main(int argc, char *argv[]) {
    // Convert command-line arguments to vector of strings
    std::vector<std::string> args = GetCommandLineArguments(argv);
    std::string volumeDescription;

    Point3f min, max, p0 = Point3f(0, 0, 0), p1 = Point3f(0.01f, 0.01f, 0.01f);
    bool minAssigned = false, maxAssigned = false;
    Float rate = 50;
    std::string outFilename;
    for (auto iter = args.begin(); iter != args.end(); ++iter) {

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        std::string tempString;

        if (ParseArg(&iter, args.end(), "min", &tempString, onError)) {
            std::vector<Float> c = SplitStringToFloats(tempString, ',');
            if (c.size() != 3) {
                usage("Didn't find three values after --min");
                return 1;
            }
            minAssigned = true;
            min = Point3f(c[0], c[1], c[2]);
        } else if (ParseArg(&iter, args.end(), "max", &tempString, onError)) {
            std::vector<Float> c = SplitStringToFloats(tempString, ',');
            if (c.size() != 3) {
                usage("Didn't find three values after --max");
                return 1;
            }
            maxAssigned = true;
            max = Point3f(c[0], c[1], c[2]);
        } else if (ParseArg(&iter, args.end(), "p0", &tempString, onError)) {
            std::vector<Float> c = SplitStringToFloats(tempString, ',');
            if (c.size() != 3) {
                usage("Didn't find three values after --p0");
                return 1;
            }
            p0 = Point3f(c[0], c[1], c[2]);
        } else if (ParseArg(&iter, args.end(), "p1", &tempString, onError)) {
            std::vector<Float> c = SplitStringToFloats(tempString, ',');
            if (c.size() != 3) {
                usage("Didn't find three values after --p1");
                return 1;
            }
            p1 = Point3f(c[0], c[1], c[2]);
        } else if (ParseArg(&iter, args.end(), "rate", &rate, onError)
                   || ParseArg(&iter, args.end(), "outfile", &outFilename, onError)) {
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
    if (!minAssigned) {
        usage("Didn't find a value for --min");
        return 1;
    }
    if (!maxAssigned) {
        usage("Didn't find a value for --max");
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

    // Write information identical for all drops
    out << R"(AttributeBegin
    MediumInterface "waterAbsorbance" ""
    NamedMaterial "water"
)";

    // Create medium to sample from
    Allocator alloc = Allocator();
    Transform renderFromMedium = Transform(SquareMatrix<4>::Diag(1, 1, 1, 1)); // identity matrix
    Spectrum sigma_a = Spectrum();
    Spectrum sigma_s = Spectrum();
    Float shapeParameterInv = DotMedium::CalculateDsdInvShapeParameter(rate);
    Float scaleParameter = DotMedium::CalculateDsdScaleParameterCentimeterRadius(rate);
    Bounds3f mediumBounds = Bounds3f(p0, p1);
    DotMedium *medium = alloc.new_object<DotMedium>(
            mediumBounds, renderFromMedium, sigma_a, sigma_s, 1, 0, shapeParameterInv, scaleParameter, 0xCA75 & 0xD095, alloc);
    Point3f minInMedium = Point3f(mediumBounds.Offset(min));
    Point3f maxInMedium = Point3f(mediumBounds.Offset(max));
    Float maxDimension = mediumBounds.Diagonal()[mediumBounds.MaxDimension()];
    int minX = std::floor(minInMedium.x);
    int minY = std::floor(minInMedium.y);
    int minZ = std::floor(minInMedium.z);
    int maxX = std::ceil(maxInMedium.x);
    int maxY = std::ceil(maxInMedium.y);
    int maxZ = std::ceil(maxInMedium.z);
    int dropCount = 0;
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
                dropCount++;
                // Calculate stuff about the drop
                Float radius = medium->cellDropRadius(cellX, cellY, cellZ);
                Vector3f center = medium->cellDropPosition(cellX, cellY, cellZ, radius);
                Float dropMoveDistance = medium->dropMoveDistanceInOneFrame(radius);
                // Write the drop to file
                Vector3f start = mediumBounds.OffsetReverse(Point3f(center.x, center.y, center.z));
                Vector3f end = mediumBounds.OffsetReverse(Point3f(center.x, center.y + dropMoveDistance, center.z));
                out << "AttributeBegin\n"
                    << "    ActiveTransform StartTime\n"
                    << "        Translate " << start.x << " " << start.y << " " << start.z << " \n"
                    << "    ActiveTransform EndTime\n"
                    << "        Translate " << end.x << " " << end.y << " " << end.z << " \n"
                    << "    ActiveTransform All\n"
                    << R"(    Shape "sphere" "float radius" [)" << radius * maxDimension << "] \n"
                    << "AttributeEnd\n";
            }
        }
    }

    // End file
    out << R"(    MediumInterface "" ""
AttributeEnd
)";
    out.close();

    Vector3f span = maxInMedium - minInMedium;
    Float volume = span.x * span.y * span.z;
    fprintf(stderr, "Wrote %i drops to a file in a volume of %f cm^3, for %f drops per cm^3", 
            dropCount, volume, Float(dropCount) / volume);
    return 0;
}