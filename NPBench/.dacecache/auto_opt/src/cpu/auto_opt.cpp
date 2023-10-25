/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct auto_opt_t {
    double * __restrict__ __0___tmp16;
    double * __restrict__ __0___tmp30;
    double * __restrict__ __0___tmp41;
    double * __restrict__ __0___tmp66;
    double * __restrict__ __0_ccol;
    double * __restrict__ __0_data_col;
    double * __restrict__ __0_cs;
    double * __restrict__ __0_bcol;
    double * __restrict__ __0_acol;
};

void __program_auto_opt_internal(auto_opt_t *__state, double * __restrict__ u_pos, double * __restrict__ u_stage, double * __restrict__ utens, double * __restrict__ utens_stage, double * __restrict__ wcon, long long I, long long J, long long K, double dtr_stage)
{
    double *dcol;
    dcol = new double DACE_ALIGN(64)[((I * J) * K)];
    long long k;




    for (k = 0; (k < 1); k = (k + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __tmp8[1]  DACE_ALIGN(64);
                        double __tmp69[1]  DACE_ALIGN(64);
                        double __tmp70[1]  DACE_ALIGN(64);
                        double __tmp71[1]  DACE_ALIGN(64);
                        double __tmp72[1]  DACE_ALIGN(64);
                        double __tmp73[1]  DACE_ALIGN(64);
                        double __tmp74[1]  DACE_ALIGN(64);
                        double __tmp88[1]  DACE_ALIGN(64);
                        double __tmp89[1]  DACE_ALIGN(64);
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

                            __tmp71[0] = __out;
                        }
                        {
                            double __in1 = __tmp71[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp69[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp69, __state->__0_cs + ((J * __i0) + __i1), 1);
                        {
                            double __in1 = __tmp69[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_USub_)
                            __out = (- __in1);
                            ///////////////////

                            __tmp8[0] = __out;
                        }
                        {
                            double __in1 = __tmp71[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp70[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp70, __tmp88, 1);
                        {
                            double __in2 = __tmp70[0];
                            double __in1 = dtr_stage;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __tmp89[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp89, __state->__0_bcol + ((J * __i0) + __i1), 1);
                        {
                            double __in2 = __tmp89[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (1.0 / __in2);
                            ///////////////////

                            __tmp74[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp88, __state->__0_ccol + ((((J * K) * __i0) + (K * __i1)) + k), 1);
                        {
                            double __in2 = __tmp74[0];
                            double __in1 = __tmp88[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __state->__0___tmp16[((J * __i0) + __i1)] = __out;
                        }
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
                            double __in1 = __tmp8[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __tmp73[0] = __out;
                        }
                        double __s2_n8__out_n9IN___tmp11;
                        {
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in1 = dtr_stage;
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
                            double __in2 = __tmp73[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __tmp72[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp72, dcol + ((((J * K) * __i0) + (K * __i1)) + k), 1);
                        {
                            double __in2 = __tmp74[0];
                            double __in1 = __tmp72[0];
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

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __state->__0___tmp16, __state->__0_ccol + k, I, J, (J * K), J, 1, K);

        }

    }



    for (k = 1; (k < (K - 1)); k = (k + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __tmp75[1]  DACE_ALIGN(64);
                        double __tmp76[1]  DACE_ALIGN(64);
                        double __tmp77[1]  DACE_ALIGN(64);
                        double __tmp78[1]  DACE_ALIGN(64);
                        double __tmp79[1]  DACE_ALIGN(64);
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

                            __tmp77[0] = __out;
                        }
                        {
                            double __in1 = __tmp77[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp76[0] = __out;
                        }
                        double __s5_n34__out_n35IN___tmp28;
                        {
                            double __in1 = __tmp76[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_USub_)
                            __out = (- __in1);
                            ///////////////////

                            __s5_n34__out_n35IN___tmp28 = __out;
                        }
                        {
                            double __in1 = __tmp77[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp75[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp75, __state->__0_acol + ((J * __i0) + __i1), 1);
                        double __s5_n28__out_n29IN___tmp26;
                        {
                            double __in1 = dtr_stage;
                            double __in2 = __tmp75[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __s5_n28__out_n29IN___tmp26 = __out;
                        }
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

                            __tmp79[0] = __out;
                        }
                        {
                            double __in1 = __tmp79[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __state->__0_cs[((J * __i0) + __i1)] = __out;
                        }
                        {
                            double __in1 = __tmp79[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp78[0] = __out;
                        }

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp78, __state->__0_ccol + ((((J * K) * __i0) + (K * __i1)) + k), 1);
                        {
                            const double __in1 = __s5_n28__out_n29IN___tmp26;
                            double __in2 = __tmp78[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __state->__0_bcol[((J * __i0) + __i1)] = __out;
                        }
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
                        {
                            const double __in1 = __s5_n34__out_n35IN___tmp28;
                            const double __in2 = __s5_n36__out_n37IN___tmp29;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __state->__0___tmp30[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __tmp43[1]  DACE_ALIGN(64);
                        double __tmp80[1]  DACE_ALIGN(64);
                        double __tmp81[1]  DACE_ALIGN(64);
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
                            double __in1 = __state->__0_cs[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __s6_n3__out_n4IN___tmp32 = __out;
                        }
                        {
                            const double __in2 = __s6_n3__out_n4IN___tmp32;
                            double __in1 = __state->__0___tmp30[((J * __i0) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __tmp81[0] = __out;
                        }
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
                            double __in2 = __tmp81[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                        double __s6_n35__out_n36IN___tmp42;
                        {
                            double __in2 = __state->__0_acol[((J * __i0) + __i1)];
                            double __in1 = dcol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
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

                            __tmp43[0] = __out;
                        }
                        double __s6_n21__out_n22IN___tmp38;
                        {
                            double __in2 = __state->__0_acol[((J * __i0) + __i1)];
                            double __in1 = __state->__0_ccol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
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
                            double __in1 = __state->__0_bcol[((J * __i0) + __i1)];
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

                            __tmp80[0] = __out;
                        }
                        {
                            double __in2 = __tmp80[0];
                            double __in1 = __state->__0_ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __state->__0___tmp41[((J * __i0) + __i1)] = __out;
                        }
                        {
                            double __in1 = __tmp43[0];
                            double __in2 = __tmp80[0];
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

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __state->__0___tmp41, __state->__0_ccol + k, I, J, (J * K), J, 1, K);

        }

    }



    for (k = (K - 1); (k < K); k = (k + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __tmp61[1]  DACE_ALIGN(64);
                        double __tmp82[1]  DACE_ALIGN(64);
                        double __tmp83[1]  DACE_ALIGN(64);
                        double __tmp84[1]  DACE_ALIGN(64);
                        double __tmp85[1]  DACE_ALIGN(64);
                        double __tmp86[1]  DACE_ALIGN(64);
                        double __tmp87[1]  DACE_ALIGN(64);
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

                            __tmp83[0] = __out;
                        }
                        {
                            double __in1 = __tmp83[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp84[0] = __out;
                        }
                        double __s9_n20__out_n21IN___tmp50;
                        {
                            double __in1 = __tmp84[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_USub_)
                            __out = (- __in1);
                            ///////////////////

                            __s9_n20__out_n21IN___tmp50 = __out;
                        }
                        {
                            double __in1 = __tmp83[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * 0.5);
                            ///////////////////

                            __tmp86[0] = __out;
                        }
                        {
                            double __in1 = dtr_stage;
                            double __in2 = __tmp86[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __tmp87[0] = __out;
                        }
                        double __s9_n37__out_n38IN___tmp57;
                        {
                            double __in1 = __state->__0_ccol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = __tmp86[0];
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
                            double __in1 = __tmp87[0];
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

                            __tmp85[0] = __out;
                        }
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
                        {
                            const double __in1 = __s9_n20__out_n21IN___tmp50;
                            const double __in2 = __s9_n22__out_n23IN___tmp51;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __tmp82[0] = __out;
                        }
                        double __s9_n27__out_n28IN___tmp53;
                        {
                            double __in2 = u_pos[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in1 = dtr_stage;
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
                            double __in2 = __tmp82[0];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            dcol[((((J * K) * __i0) + (K * __i1)) + k)] = __out;
                        }
                        double __s9_n44__out_n45IN___tmp60;
                        {
                            double __in1 = dcol[(((((J * K) * __i0) + (K * __i1)) + k) - 1)];
                            double __in2 = __tmp86[0];
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

                            __tmp61[0] = __out;
                        }
                        {
                            double __in1 = __tmp61[0];
                            double __in2 = __tmp85[0];
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

        }

    }



    for (k = (K - 1); (k > (K - 2)); k = (k + (- 1))) {
        {


            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            dcol + k, __state->__0_data_col, I, (J * K), J, J, K, 1);
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

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < I; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < J; __i1 += 1) {
                        double __s15_n1__out_n2IN___tmp65;
                        {
                            double __in1 = __state->__0_ccol[((((J * K) * __i0) + (K * __i1)) + k)];
                            double __in2 = __state->__0_data_col[((J * __i0) + __i1)];
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

                            __state->__0___tmp66[((J * __i0) + __i1)] = __out;
                        }
                    }
                }
            }

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            __state->__0___tmp66, dcol + k, I, J, (J * K), J, 1, K);

            dace::CopyNDDynamic<double, 1, false, 2>::Dynamic::Copy(
            dcol + k, __state->__0_data_col, I, (J * K), J, J, K, 1);
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

        }

    }

    delete[] dcol;
}

DACE_EXPORTED void __program_auto_opt(auto_opt_t *__state, double * __restrict__ u_pos, double * __restrict__ u_stage, double * __restrict__ utens, double * __restrict__ utens_stage, double * __restrict__ wcon, long long I, long long J, long long K, double dtr_stage)
{
    __program_auto_opt_internal(__state, u_pos, u_stage, utens, utens_stage, wcon, I, J, K, dtr_stage);
}

DACE_EXPORTED auto_opt_t *__dace_init_auto_opt(long long I, long long J, long long K)
{
    int __result = 0;
    auto_opt_t *__state = new auto_opt_t;


    __state->__0___tmp16 = new double DACE_ALIGN(64)[(I * J)];
    __state->__0___tmp30 = new double DACE_ALIGN(64)[(I * J)];
    __state->__0___tmp41 = new double DACE_ALIGN(64)[(I * J)];
    __state->__0___tmp66 = new double DACE_ALIGN(64)[(I * J)];
    __state->__0_ccol = new double DACE_ALIGN(64)[((I * J) * K)];
    __state->__0_data_col = new double DACE_ALIGN(64)[(I * J)];
    __state->__0_cs = new double DACE_ALIGN(64)[(I * J)];
    __state->__0_bcol = new double DACE_ALIGN(64)[(I * J)];
    __state->__0_acol = new double DACE_ALIGN(64)[(I * J)];

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_auto_opt(auto_opt_t *__state)
{
    int __err = 0;
    delete[] __state->__0___tmp16;
    delete[] __state->__0___tmp30;
    delete[] __state->__0___tmp41;
    delete[] __state->__0___tmp66;
    delete[] __state->__0_ccol;
    delete[] __state->__0_data_col;
    delete[] __state->__0_cs;
    delete[] __state->__0_bcol;
    delete[] __state->__0_acol;
    delete __state;
    return __err;
}

