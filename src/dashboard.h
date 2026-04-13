#ifndef DASHBOARD_H
#define DASHBOARD_H

#include "../include/option_params.h"

/* Launch the interactive ncurses terminal dashboard */
void launch_dashboard(OptionParams *initial_params, int num_paths,
                      int num_steps, int block_size);

#endif /* DASHBOARD_H */
