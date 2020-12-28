#ifndef IO_HPP
#define IO_HPP

#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <fmt/core.h>
#include <fmt/ostream.h>

struct Dataset {
    unsigned int cardinality;
    unsigned int dimensions;
    float* pointsRow;
    float* pointsCol;

    ~Dataset() {
        delete[] pointsRow;
        delete[] pointsCol;
    }
};


std::vector<float> split(const std::string& s, char delimiter) {
    std::stringstream ss(s);
    std::string item;
    std::vector<float> elements;
    while (std::getline(ss, item, delimiter)) {
        elements.push_back(std::stof(item));
    }
    return elements;
}

Dataset* readDataset(std::string& path, unsigned int cardinality, unsigned int dimensions) {
    auto d = new Dataset;
    d->cardinality = cardinality;
    d->dimensions = dimensions;

    std::ifstream infile;
    std::string line;
    infile.open(path.c_str());
    if (infile.fail()) {
        fmt::print("{}\n", "Wrong input dataset! Exiting...");
        exit(1);
    }

    d->pointsRow = new float[d->cardinality * d->dimensions];

    unsigned int i = 0;

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.empty()) continue;

        std::vector<float> v = split(line, ',');

        for(auto& point : v) {
            d->pointsRow[i++] = point;
        }
    }

    infile.close();

    // Additionally, store dataset in column format to achieve better memory access in GPU
    d->pointsCol = new float[d->cardinality * d->dimensions];

    for (i = 0; i < d->cardinality; i++) {
        for (unsigned int j = 0; j < d->dimensions; j++) {
            d->pointsCol[(j * d->cardinality) + i] = d->pointsRow[(i * d->dimensions) + j];
        }
    }

    return d;
}

void writeResult(std::vector<unsigned int> counts, std::string& output) {
    std::ofstream file;
    file.open(output.c_str());

    for (unsigned int i = 0; i < counts.size(); ++i) {
        fmt::print(file, "Run({}): {}\n", i, counts[i]);
    }

    file.close();
}

#endif //IO_HPP
