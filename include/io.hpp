#ifndef IO_HPP
#define IO_HPP

#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <fmt/core.h>
#include <fmt/ostream.h>


template<typename T>
std::vector<T> split(const std::string& s, char delimiter) {
    std::stringstream ss(s);
    std::string item;
    std::vector<T> elements;
    while (std::getline(ss, item, delimiter)) {
        elements.push_back(typeid(T) == typeid(float) ? std::stof(item) : std::stod(item));
    }
    return elements;
}

template<class T>
struct Dataset {
    unsigned int cardinality;
    unsigned int dimensions;
    T* pointsRow;
    T* pointsCol;
    T* weights;

    Dataset(std::string& path, unsigned int cardinality, unsigned int dimensions, bool skipHeader) :
            cardinality(cardinality), dimensions(dimensions) {
        std::ifstream infile;
        std::string line;
        infile.open(path.c_str());
        if (infile.fail()) {
            fmt::print("{}\n", "Wrong input dataset! Exiting...");
            exit(1);
        }

        this->pointsRow = new T[this->cardinality * this->dimensions];

        unsigned int i = 0;

        if (skipHeader) std::getline(infile, line);

        while (!infile.eof()) {
            std::getline(infile, line);
            if (line.empty()) continue;
            std::vector<T> v = split<T>(line, ',');
            for (auto& point: v) {
                this->pointsRow[i++] = point;
            }
        }

        infile.close();

        // Additionally, store dataset in column format to achieve better memory access in GPU
        this->pointsCol = new T[this->cardinality * this->dimensions];

        for (i = 0; i < this->cardinality; i++) {
            for (unsigned int j = 0; j < this->dimensions; j++) {
                this->pointsCol[(j * this->cardinality) + i] = this->pointsRow[(i * this->dimensions) + j];
            }
        }
    }

    void readWeights(std::string& path) {
        weights = new T[this->dimensions]{1.00};
        std::ifstream infile;
        std::string line;
        infile.open(path.c_str());

        std::getline(infile, line);

        std::vector<T> v = split<T>(line, ',');
        for (unsigned int i = 0; i < v.size(); ++i) {
            weights[i] = v[i];
        }

        infile.close();
    }

    ~Dataset() {
        delete[] pointsRow;
        delete[] pointsCol;
        delete[] weights;
    }
};

std::set<unsigned int> readGroundTruth(std::string& path) {
    std::set<unsigned int> ids;
    std::ifstream infile;
    std::string line;
    infile.open(path.c_str());

    std::getline(infile, line);

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.empty()) continue;
        ids.insert(std::stoi(line));
    }

    infile.close();
    return ids;
}


void writeCountsResult(std::vector<unsigned int> counts,
                       double threshold,
                       unsigned int k,
                       std::string& output) {
    std::ofstream file;
    file.open(output.c_str());
    fmt::print(file, "Radius={}, k={}\n", threshold, k);

    for (unsigned int i = 0; i < counts.size(); ++i) {
        fmt::print(file, "Run({}): {}\n", i, counts[i]);
    }

    file.close();
}

void writeOutliersResult(std::vector<unsigned int> outliers,
                         double threshold,
                         unsigned int k,
                         std::string& output) {
    std::ofstream file;
    file.open(output.c_str());
    fmt::print(file, "Radius={}, k={}\n", threshold, k);

    for (unsigned int i = 0; i < outliers.size(); ++i) {
        fmt::print(file, "{}\n", outliers[i]);
    }

    file.close();
}

#endif //IO_HPP
