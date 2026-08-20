#pragma once

#include <cstdint>

struct config_st {
    int transpose_matrix_flag;
    short extension_penalty;
    int all_vs_all;
    const char *seq1_file_name;
    const char *seq2_file_name;
    const char *seq1_cli;
    const char *seq2_cli;
    const char *mat_name;
    const char *pos_weights;
    int min_score;
    int doSubopt;
    double max_energy;
    int filter_e;
    int weighted_positions;
    int vicinity;
    char printShort;
    int force_start_val;
    std::uint32_t tblen;
};
