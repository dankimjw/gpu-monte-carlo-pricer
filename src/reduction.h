#ifndef REDUCTION_H
#define REDUCTION_H

void reduce_payoffs(float *d_payoffs, int num_paths,
                    float *mean_out, float *variance_out);

#endif /* REDUCTION_H */
