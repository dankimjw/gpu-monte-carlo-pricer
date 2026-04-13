#ifndef GL_DASHBOARD_H
#define GL_DASHBOARD_H

#include "../include/option_params.h"

/* Launch the OpenGL graphical dashboard */
void launch_gl_dashboard(OptionParams *initial_params, int num_paths,
                         int num_steps, int block_size);

#endif /* GL_DASHBOARD_H */
