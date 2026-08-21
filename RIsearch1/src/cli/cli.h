#pragma once

#include <cstdint>

#include "cli/config.h"

/**
 * print help / documentation for the user
 */
[[noreturn]] void usage(const char* progname);


/* getopt hands every option in as text. A length has no reading below zero, and
   it is kept unsigned, where a negative would wrap into an enormous one. Refuse
   it where it is read rather than let it through. */
std::uint32_t parse_length_arg(const char* text, char option);


/**
 * Retrieve config arguments from user and assign to config.
 *
 * Function exists the program if impossible config is provided.
 */
void getArgs(int argc, char* argv[], config_st& config);


inline bool uses_force_start(const config_st& config)
{
    return config.weighted_positions || config.force_start_val >= 0;
}

/**
 * Check if force_start config is possible, exist otherwise.
 */
void validate_force_start_config(const config_st& config);
