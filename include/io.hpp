#ifndef IO_HPP
#define IO_HPP

#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
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

    Dataset(std::string& path, unsigned int cardinality, unsigned int dimensions) : cardinality(cardinality),
                                                                                    dimensions(dimensions) {
        std::ifstream infile;
        std::string line;
        infile.open(path.c_str());
        if (infile.fail()) {
            fmt::print("{}\n", "Wrong input dataset! Exiting...");
            exit(1);
        }

        this->pointsRow = new T[this->cardinality * this->dimensions];

        unsigned int i = 0;

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


void writeCountsResult(std::vector<unsigned int> counts, std::string& output) {
    std::ofstream file;
    file.open(output.c_str());

    for (unsigned int i = 0; i < counts.size(); ++i) {
        fmt::print(file, "Run({}): {}\n", i, counts[i]);
    }

    file.close();
}

void writeOutliersResult(std::vector<unsigned int> outliers, std::string& output) {
    std::ofstream file;
    file.open(output.c_str());

    for (unsigned int i = 0; i < outliers.size(); ++i) {
        fmt::print(file, "{}\n", outliers[i]);
    }

    file.close();
}

#endif //IO_HPP
