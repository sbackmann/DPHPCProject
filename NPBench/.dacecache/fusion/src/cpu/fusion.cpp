/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct fusion_t {

};

void __program_fusion_internal(fusion_t *__state, double * __restrict__ u_pos, double * __restrict__ u_stage, double * __restrict__ utens, double * __restrict__ utens_stage, double * __restrict__ wcon, long long I, long long J, long long K, double dtr_stage)
{
    double *__tmp8;
    __tmp8 = new double DACE_ALIGN(64)[(I * J)];
    double *__tmp30;
    __tmp30 = new double DACE_ALIGN(64)[(I * J)];
    double *ccol;
    ccol = new double DACE_ALIGN(64)[((I * J) * K)];
    double *dcol;
    dcol = new double DACE_ALIGN(64)[((I * J) * K)];
    double *data_col;
    data_col = new double DACE_ALIGN(64)[(I * J)];
    double *gcv;
    gcv = new double DACE_ALIGN(64)[(I * J)];
    double *cs;
    cs = new double DACE_ALIGN(64)[(I * J)];
    double *bcol;
    bcol = new double DACE_ALIGN(64)[(I * J)];
    double *correction_term;
    correction_term = new double DACE_ALIGN(64)[(I * J)];
    double *divided;
    divided = new double DACE_ALIGN(64)[(I * J)];
    double *gav;
    gav = new double DACE_ALIGN(64)[(I * J)];
    double *as_;
    as_ = new double DACE_ALIGN(64)[(I * J)];
    double *acol;
    acol = new double DACE_ALIGN(64)[(I * J)];
    long long k;




    for (k = 0; (k < 1); k = (k + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s1_n1__out_n2IN___tmp3;
                        {
                            double __in1 = wcon[(((((J * K) * (__i0 + 1)) + (K * __i1)) + k) + 1)];
                            double __in2 = wcon[(((((J * K) * __i0) + (K * __i1)) + k) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s1_n1__out_n2IN___tmp3 = __out;
                        }
                        {
                            const double __in2 = __s1_n1__out_n2IN___tmp3;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (0.25 * __in2);
                            ///////////////////

                            gcv[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gcv[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            cs[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = cs[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_USub_)
                            __out = (- __in1);
                            ///////////////////

                            __tmp8[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gcv[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            ccol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = dtr_stage;
                            double __in2 = ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            bcol[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {
            double *__tmp16;
            __tmp16 = new double DACE_ALIGN(64)[(I * J)];
            double *__tmp17;
            __tmp17 = new double DACE_ALIGN(64)[(I * J)];

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s2_n1__out_n2IN___tmp9;
                        {
                            double __in1 = u_stage[(((((J * K) * __i0) + (K * __i1)) + k) + 1)];
                            double __in2 = u_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s2_n1__out_n2IN___tmp9 = __out;
                        }
                        {
                            const double __in2 = __s2_n1__out_n2IN___tmp9;
                            double __in1 = __tmp8[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            correction_term[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s2_n8__out_n9IN___tmp11;
                        {
                            double __in1 = dtr_stage;
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s2_n8__out_n9IN___tmp11 = __out;
                        }
                        double __s2_n11__out_n12IN___tmp12;
                        {
                            const double __in1 = __s2_n8__out_n9IN___tmp11;
                            double __in2 = utens[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s2_n11__out_n12IN___tmp12 = __out;
                        }
                        double __s2_n13__out_n14IN___tmp13;
                        {
                            const double __in1 = __s2_n11__out_n12IN___tmp12;
                            double __in2 = utens_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s2_n13__out_n14IN___tmp13 = __out;
                        }
                        {
                            const double __in1 = __s2_n13__out_n14IN___tmp13;
                            double __in2 = correction_term[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in2 = bcol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (1.0 / __in2);
                            ///////////////////

                            divided[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = dcol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = divided[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __tmp17[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __tmp17, dcol + k, I, J, (J * K), J, 1, K);
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = divided[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __tmp16[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __tmp16, ccol + k, I, J, (J * K), J, 1, K);
            delete[] __tmp16;
            delete[] __tmp17;

        }

    }



    for (k = 1; (k < (K - 1)); k = (k + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s5_n1__out_n2IN___tmp18;
                        {
                            double __in1 = wcon[((((J * K) * (__i0 + 1)) + (K * __i1)) + k)];
                            double __in2 = wcon[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s5_n1__out_n2IN___tmp18 = __out;
                        }
                        {
                            const double __in2 = __s5_n1__out_n2IN___tmp18;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (-0.25 * __in2);
                            ///////////////////

                            gav[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gav[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            as_[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gav[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            acol[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s5_n7__out_n8IN___tmp20;
                        {
                            double __in1 = wcon[(((((J * K) * (__i0 + 1)) + (K * __i1)) + k) + 1)];
                            double __in2 = wcon[(((((J * K) * __i0) + (K * __i1)) + k) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s5_n7__out_n8IN___tmp20 = __out;
                        }
                        {
                            const double __in2 = __s5_n7__out_n8IN___tmp20;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (0.25 * __in2);
                            ///////////////////

                            gcv[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gcv[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            cs[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gcv[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            ccol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s5_n28__out_n29IN___tmp26;
                        {
                            double __in1 = dtr_stage;
                            double __in2 = acol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s5_n28__out_n29IN___tmp26 = __out;
                        }
                        {
                            const double __in1 = __s5_n28__out_n29IN___tmp26;
                            double __in2 = ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            bcol[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s5_n36__out_n37IN___tmp29;
                        {
                            double __in1 = u_stage[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = u_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s5_n36__out_n37IN___tmp29 = __out;
                        }
                        double __s5_n34__out_n35IN___tmp28;
                        {
                            double __in1 = as_[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_USub_)
                            __out = (- __in1);
                            ///////////////////

                            __s5_n34__out_n35IN___tmp28 = __out;
                        }
                        {
                            const double __in1 = __s5_n34__out_n35IN___tmp28;
                            const double __in2 = __s5_n36__out_n37IN___tmp29;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __tmp30[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {
            double *__tmp41;
            __tmp41 = new double DACE_ALIGN(64)[(I * J)];
            double *__tmp43;
            __tmp43 = new double DACE_ALIGN(64)[(I * J)];

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s6_n1__out_n2IN___tmp31;
                        {
                            double __in1 = u_stage[(((((J * K) * __i0) + (K * __i1)) + k) + 1)];
                            double __in2 = u_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s6_n1__out_n2IN___tmp31 = __out;
                        }
                        double __s6_n3__out_n4IN___tmp32;
                        {
                            const double __in2 = __s6_n1__out_n2IN___tmp31;
                            double __in1 = cs[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s6_n3__out_n4IN___tmp32 = __out;
                        }
                        {
                            const double __in2 = __s6_n3__out_n4IN___tmp32;
                            double __in1 = __tmp30[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            correction_term[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s6_n10__out_n11IN___tmp34;
                        {
                            double __in1 = dtr_stage;
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s6_n10__out_n11IN___tmp34 = __out;
                        }
                        double __s6_n13__out_n14IN___tmp35;
                        {
                            const double __in1 = __s6_n10__out_n11IN___tmp34;
                            double __in2 = utens[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s6_n13__out_n14IN___tmp35 = __out;
                        }
                        double __s6_n15__out_n16IN___tmp36;
                        {
                            const double __in1 = __s6_n13__out_n14IN___tmp35;
                            double __in2 = utens_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s6_n15__out_n16IN___tmp36 = __out;
                        }
                        {
                            const double __in1 = __s6_n15__out_n16IN___tmp36;
                            double __in2 = correction_term[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s6_n35__out_n36IN___tmp42;
                        {
                            double __in1 = dcol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = acol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s6_n35__out_n36IN___tmp42 = __out;
                        }
                        {
                            const double __in2 = __s6_n35__out_n36IN___tmp42;
                            double __in1 = dcol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __tmp43[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s6_n21__out_n22IN___tmp38;
                        {
                            double __in1 = ccol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = acol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s6_n21__out_n22IN___tmp38 = __out;
                        }
                        double __s6_n24__out_n25IN___tmp39;
                        {
                            const double __in2 = __s6_n21__out_n22IN___tmp38;
                            double __in1 = bcol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s6_n24__out_n25IN___tmp39 = __out;
                        }
                        {
                            const double __in2 = __s6_n24__out_n25IN___tmp39;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (1.0 / __in2);
                            ///////////////////

                            divided[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = divided[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __tmp41[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __tmp41, ccol + k, I, J, (J * K), J, 1, K);
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = __tmp43[((J * __i0) + __i1)];
                            double __in2 = divided[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            delete[] __tmp41;
            delete[] __tmp43;

        }

    }



    for (k = (K - 1); (k < K); k = (k + 1)) {
        {
            double *__tmp61;
            __tmp61 = new double DACE_ALIGN(64)[(I * J)];

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s9_n1__out_n2IN___tmp45;
                        {
                            double __in1 = wcon[((((J * K) * (__i0 + 1)) + (K * __i1)) + k)];
                            double __in2 = wcon[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s9_n1__out_n2IN___tmp45 = __out;
                        }
                        {
                            const double __in2 = __s9_n1__out_n2IN___tmp45;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (-0.25 * __in2);
                            ///////////////////

                            gav[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gav[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            as_[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = gav[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            acol[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = dtr_stage;
                            double __in2 = acol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            bcol[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s9_n22__out_n23IN___tmp51;
                        {
                            double __in1 = u_stage[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = u_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s9_n22__out_n23IN___tmp51 = __out;
                        }
                        double __s9_n20__out_n21IN___tmp50;
                        {
                            double __in1 = as_[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_USub_)
                            __out = (- __in1);
                            ///////////////////

                            __s9_n20__out_n21IN___tmp50 = __out;
                        }
                        {
                            const double __in1 = __s9_n20__out_n21IN___tmp50;
                            const double __in2 = __s9_n22__out_n23IN___tmp51;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            correction_term[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s9_n27__out_n28IN___tmp53;
                        {
                            double __in1 = dtr_stage;
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s9_n27__out_n28IN___tmp53 = __out;
                        }
                        double __s9_n29__out_n30IN___tmp54;
                        {
                            const double __in1 = __s9_n27__out_n28IN___tmp53;
                            double __in2 = utens[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s9_n29__out_n30IN___tmp54 = __out;
                        }
                        double __s9_n31__out_n32IN___tmp55;
                        {
                            const double __in1 = __s9_n29__out_n30IN___tmp54;
                            double __in2 = utens_stage[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __s9_n31__out_n32IN___tmp55 = __out;
                        }
                        {
                            const double __in1 = __s9_n31__out_n32IN___tmp55;
                            double __in2 = correction_term[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s9_n44__out_n45IN___tmp60;
                        {
                            double __in1 = dcol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = acol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s9_n44__out_n45IN___tmp60 = __out;
                        }
                        {
                            const double __in2 = __s9_n44__out_n45IN___tmp60;
                            double __in1 = dcol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __tmp61[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s9_n37__out_n38IN___tmp57;
                        {
                            double __in1 = ccol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = acol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s9_n37__out_n38IN___tmp57 = __out;
                        }
                        double __s9_n39__out_n40IN___tmp58;
                        {
                            const double __in2 = __s9_n37__out_n38IN___tmp57;
                            double __in1 = bcol[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s9_n39__out_n40IN___tmp58 = __out;
                        }
                        {
                            const double __in2 = __s9_n39__out_n40IN___tmp58;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (1.0 / __in2);
                            ///////////////////

                            divided[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        {
                            double __in1 = __tmp61[((J * __i0) + __i1)];
                            double __in2 = divided[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            delete[] __tmp61;

        }

    }



    for (k = (K - 1); (k > (K - 2)); k = (k + (- 1))) {
        {


            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            dcol + k, data_col, I, (J * K), J, J, K, 1);
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s12_n3__out_n4IN___tmp63;
                        {
                            double __in1 = dcol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s12_n3__out_n4IN___tmp63 = __out;
                        }
                        {
                            const double __in2 = __s12_n3__out_n4IN___tmp63;
                            double __in1 = dtr_stage;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            utens_stage[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }

        }

    }



    for (k = (K - 2); (k > -1); k = (k + (- 1))) {
        {
            double *__tmp66;
            __tmp66 = new double DACE_ALIGN(64)[(I * J)];

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s15_n1__out_n2IN___tmp65;
                        {
                            double __in1 = ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = data_col[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s15_n1__out_n2IN___tmp65 = __out;
                        }
                        {
                            const double __in2 = __s15_n1__out_n2IN___tmp65;
                            double __in1 = dcol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __tmp66[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __tmp66, dcol + k, I, J, (J * K), J, 1, K);

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            dcol + k, data_col, I, (J * K), J, J, K, 1);
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s15_n11__out_n12IN___tmp67;
                        {
                            double __in1 = dcol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s15_n11__out_n12IN___tmp67 = __out;
                        }
                        {
                            const double __in2 = __s15_n11__out_n12IN___tmp67;
                            double __in1 = dtr_stage;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            utens_stage[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                    }
                }
            }
            delete[] __tmp66;

        }

    }

    delete[] __tmp8;
    delete[] __tmp30;
    delete[] ccol;
    delete[] dcol;
    delete[] data_col;
    delete[] gcv;
    delete[] cs;
    delete[] bcol;
    delete[] correction_term;
    delete[] divided;
    delete[] gav;
    delete[] as_;
    delete[] acol;
}

DACE_EXPORTED void __program_fusion(fusion_t *__state, double * __restrict__ u_pos, double * __restrict__ u_stage, double * __restrict__ utens, double * __restrict__ utens_stage, double * __restrict__ wcon, long long I, long long J, long long K, double dtr_stage)
{
    __program_fusion_internal(__state, u_pos, u_stage, utens, utens_stage, wcon, I, J, K, dtr_stage);
}

DACE_EXPORTED fusion_t *__dace_init_fusion(long long I, long long J, long long K)
{
    int __result = 0;
    fusion_t *__state = new fusion_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_fusion(fusion_t *__state)
{
    int __err = 0;
    delete __state;
    return __err;
}

