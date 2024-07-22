//////////////////////////////////////////////////////////////////////
// InferenceEngine.hpp
//////////////////////////////////////////////////////////////////////

#ifndef INFERENCEENGINE_HPP
#define INFERENCEENGINE_HPP

#include <queue>
#include <vector>
#include <string>
#include "Config.hpp"
#include "SStruct.hpp"
#include "ParameterManager.hpp"
#include "Utilities.hpp"
#include "LogSpace.hpp"

//////////////////////////////////////////////////////////////////////
// class InferenceEngine
//////////////////////////////////////////////////////////////////////

template<class RealT>
class InferenceEngine
{
    const bool allow_noncomplementary;
    unsigned char char_mapping[256];
    int is_complementary[M+1][M+1];
    bool cache_initialized;
    ParameterManager<RealT> *parameter_manager;
    
    // dimensions
    int L, SIZE;
#if PROFILE
    int N, SIZE2;
#endif

    // sequence data
    std::vector<int> s, offset;
#if PROFILE
    std::vector<int> A;
    std::vector<RealT> weights;
#endif
    std::vector<int> allow_unpaired_position;
    std::vector<int> allow_unpaired, allow_paired;
    std::vector<RealT> loss_unpaired_position;
    std::vector<RealT> loss_unpaired, loss_paired;

    enum TRACEBACK_TYPE {
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
        TB_FN_HAIRPIN,
        TB_FN_SINGLE,
        TB_FN_BIFURCATION,
        TB_FE_STACKING,
        TB_FE_FN,
        TB_FC_FN,
        TB_FC_HELIX,
        TB_FC_FE,
#else
        TB_FC_HAIRPIN,
        TB_FC_SINGLE,
        TB_FC_BIFURCATION,
#endif
        TB_FM1_PAIRED,
        TB_FM1_UNPAIRED,
        TB_FM_BIFURCATION,
        TB_FM_UNPAIRED,
        TB_FM_FM1,
        TB_F5_ZERO,
        TB_F5_UNPAIRED,
        TB_F5_BIFURCATION,
        NUM_TRACEBACK_TYPES
    };
    
    // dynamic programming matrices
    std::vector<int> FCt, F5t, FMt, FM1t;            // traceback
    std::vector<RealT> FCv, F5v, FMv, FM1v;          // Viterbi  
    std::vector<RealT> FCi, F5i, FMi, FM1i;          // inside
    std::vector<RealT> FCo, F5o, FMo, FM1o;          // outside
    
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
    std::vector<int> FEt, FNt;
    std::vector<RealT> FEv, FNv;
    std::vector<RealT> FEi, FNi;
    std::vector<RealT> FEo, FNo;
#endif
    
    std::vector<RealT> posterior;

    // parameters

#if PARAMS_BASE_PAIR
    std::pair<RealT,RealT> score_base_pair[M+1][M+1];
#endif
#if PARAMS_BASE_PAIR_DIST
    std::pair<RealT,RealT> score_base_pair_dist_at_least[D_MAX_BP_DIST_THRESHOLDS];
    std::pair<RealT,RealT> cache_score_base_pair_dist[BP_DIST_LAST_THRESHOLD+1];
#endif
#if PARAMS_TERMINAL_MISMATCH
    std::pair<RealT,RealT> score_terminal_mismatch[M+1][M+1][M+1][M+1];
#endif
#if PARAMS_HAIRPIN_LENGTH
    std::pair<RealT,RealT> score_hairpin_length_at_least[D_MAX_HAIRPIN_LENGTH+1];
    std::pair<RealT,RealT> cache_score_hairpin_length[D_MAX_HAIRPIN_LENGTH+1];
#endif
#if PARAMS_HAIRPIN_3_NUCLEOTIDES
    std::pair<RealT,RealT> score_hairpin_3_nucleotides[M+1][M+1][M+1];
#endif
#if PARAMS_HAIRPIN_4_NUCLEOTIDES
    std::pair<RealT,RealT> score_hairpin_4_nucleotides[M+1][M+1][M+1][M+1];
#endif
#if PARAMS_HELIX_LENGTH
    std::pair<RealT,RealT> score_helix_length_at_least[D_MAX_HELIX_LENGTH+1];
    std::pair<RealT,RealT> cache_score_helix_length[D_MAX_HELIX_LENGTH+1];
#endif
#if PARAMS_ISOLATED_BASE_PAIR
    std::pair<RealT,RealT> score_isolated_base_pair;
#endif
#if PARAMS_INTERNAL_EXPLICIT
    std::pair<RealT,RealT> score_internal_explicit[D_MAX_INTERNAL_EXPLICIT_LENGTH+1][D_MAX_INTERNAL_EXPLICIT_LENGTH+1];
#endif
#if PARAMS_BULGE_LENGTH
    std::pair<RealT,RealT> score_bulge_length_at_least[D_MAX_BULGE_LENGTH+1];
#endif
#if PARAMS_INTERNAL_LENGTH
    std::pair<RealT,RealT> score_internal_length_at_least[D_MAX_INTERNAL_LENGTH+1];
#endif
#if PARAMS_INTERNAL_SYMMETRY
    std::pair<RealT,RealT> score_internal_symmetric_length_at_least[D_MAX_INTERNAL_SYMMETRIC_LENGTH+1];
#endif
#if PARAMS_INTERNAL_ASYMMETRY
    std::pair<RealT,RealT> score_internal_asymmetry_at_least[D_MAX_INTERNAL_ASYMMETRY+1];
#endif
#if PARAMS_BULGE_0x1_NUCLEOTIDES
    std::pair<RealT,RealT> score_bulge_0x1_nucleotides[M+1];
    std::pair<RealT,RealT> score_bulge_1x0_nucleotides[M+1];
#endif
#if PARAMS_BULGE_0x2_NUCLEOTIDES
    std::pair<RealT,RealT> score_bulge_0x2_nucleotides[M+1][M+1];
    std::pair<RealT,RealT> score_bulge_2x0_nucleotides[M+1][M+1];
#endif
#if PARAMS_BULGE_0x3_NUCLEOTIDES
    std::pair<RealT,RealT> score_bulge_0x3_nucleotides[M+1][M+1][M+1];
    std::pair<RealT,RealT> score_bulge_3x0_nucleotides[M+1][M+1][M+1];
#endif
#if PARAMS_INTERNAL_1x1_NUCLEOTIDES
    std::pair<RealT,RealT> score_internal_1x1_nucleotides[M+1][M+1];
#endif
#if PARAMS_INTERNAL_1x2_NUCLEOTIDES
    std::pair<RealT,RealT> score_internal_1x2_nucleotides[M+1][M+1][M+1];
    std::pair<RealT,RealT> score_internal_2x1_nucleotides[M+1][M+1][M+1];
#endif
#if PARAMS_INTERNAL_2x2_NUCLEOTIDES
    std::pair<RealT,RealT> score_internal_2x2_nucleotides[M+1][M+1][M+1][M+1];
#endif
#if PARAMS_HELIX_STACKING
    std::pair<RealT,RealT> score_helix_stacking[M+1][M+1][M+1][M+1];
#endif
#if PARAMS_HELIX_CLOSING
    std::pair<RealT,RealT> score_helix_closing[M+1][M+1];
#endif
#if PARAMS_MULTI_LENGTH
    std::pair<RealT,RealT> score_multi_base;
    std::pair<RealT,RealT> score_multi_unpaired;
    std::pair<RealT,RealT> score_multi_paired;
#endif
#if PARAMS_DANGLE
    std::pair<RealT,RealT> score_dangle_left[M+1][M+1][M+1];
    std::pair<RealT,RealT> score_dangle_right[M+1][M+1][M+1];
#endif
#if PARAMS_EXTERNAL_LENGTH
    std::pair<RealT,RealT> score_external_unpaired;
    std::pair<RealT,RealT> score_external_paired;
#endif
    
#if PROFILE

    // multiple sequence scoring
#if PARAMS_BASE_PAIR
    std::vector<std::pair<RealT,RealT> > profile_score_base_pair;
#endif
#if PARAMS_TERMINAL_MISMATCH
    std::vector<std::pair<RealT,RealT> > profile_score_terminal_mismatch;
#endif
#if PARAMS_HAIRPIN_3_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_hairpin_3_nucleotides;
#endif
#if PARAMS_HAIRPIN_4_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_hairpin_4_nucleotides;
#endif
#if PARAMS_BULGE_0x1_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_bulge_0x1_nucleotides;
    std::vector<std::pair<RealT,RealT> > profile_score_bulge_1x0_nucleotides;
#endif
#if PARAMS_BULGE_0x2_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_bulge_0x2_nucleotides;
    std::vector<std::pair<RealT,RealT> > profile_score_bulge_2x0_nucleotides;
#endif
#if PARAMS_BULGE_0x3_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_bulge_0x3_nucleotides;
    std::vector<std::pair<RealT,RealT> > profile_score_bulge_3x0_nucleotides;
#endif
#if PARAMS_INTERNAL_1x1_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_internal_1x1_nucleotides;
#endif
#if PARAMS_INTERNAL_1x2_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_internal_1x2_nucleotides;
    std::vector<std::pair<RealT,RealT> > profile_score_internal_2x1_nucleotides;
#endif
#if PARAMS_INTERNAL_2x2_NUCLEOTIDES
    std::vector<std::pair<RealT,RealT> > profile_score_internal_2x2_nucleotides;
#endif
#if PARAMS_HELIX_STACKING
    std::vector<std::pair<RealT,RealT> > profile_score_helix_stacking;
#endif
#if PARAMS_HELIX_CLOSING
    std::vector<std::pair<RealT,RealT> > profile_score_helix_closing;
#endif
#if PARAMS_DANGLE
    std::vector<std::pair<RealT,RealT> > profile_score_dangle_left;
    std::vector<std::pair<RealT,RealT> > profile_score_dangle_right;
#endif

#endif

    // cache
    std::pair<RealT,RealT> cache_score_single[C_MAX_SINGLE_LENGTH+1][C_MAX_SINGLE_LENGTH+1];
    std::vector<std::pair<RealT,RealT> > cache_score_helix_sums;

    void FillScores(typename std::vector<std::pair<RealT, RealT> >::iterator begin, typename std::vector<std::pair<RealT, RealT> >::iterator end, RealT value);
    void FillCounts(typename std::vector<std::pair<RealT, RealT> >::iterator begin, typename std::vector<std::pair<RealT, RealT> >::iterator end, RealT value);
    int ComputeRowOffset(int i, int N) const;
    bool IsComplementary(int i, int j) const;
    
    RealT ScoreJunctionA(int i, int j) const;
    RealT ScoreJunctionB(int i, int j) const;
    RealT ScoreBasePair(int i, int j) const;
    RealT ScoreHairpin(int i, int j) const;
    RealT ScoreHelix(int i, int j, int m) const;
    RealT ScoreSingleNucleotides(int i, int j, int p, int q) const;
    RealT ScoreSingle(int i, int j, int p, int q) const;
    
    void CountJunctionA(int i, int j, RealT value);
    void CountJunctionB(int i, int j, RealT value);
    void CountBasePair(int i, int j, RealT value);
    void CountHairpin(int i, int j, RealT value);
    void CountHelix(int i, int j, int m, RealT value);
    void CountSingleNucleotides(int i, int j, int p, int q, RealT value);
    void CountSingle(int i, int j, int p, int q, RealT value);

    int EncodeTraceback(int i, int j) const;
    std::pair<int,int> DecodeTraceback(int s) const;

    std::vector<RealT> GetCounts();
    void ClearCounts();
    void InitializeCache();
    void FinalizeCounts();
    
#if PROFILE
    void ComputeProfileScore(RealT &profile_score, const int *pos, int dimensions, std::pair<RealT,RealT> *table);
    void ConvertProfileCount(const RealT &profile_score, const int *pos, int dimensions, std::pair<RealT,RealT> *table);
#endif
    
public:

    // constructor and destructor
    InferenceEngine(bool allow_noncomplementary);
    ~InferenceEngine();

    // register params with the parameter manager
    void RegisterParameters(ParameterManager<RealT> &parameter_manager);
                            
    // load sequence
    void LoadSequence(const SStruct &sstruct);
    
    // load parameter values                        
    void LoadValues(const std::vector<RealT> &values);
    
    // load loss function
    void UseLoss(const std::vector<int> &true_mapping, RealT example_loss);

    // use constraints
    void UseConstraints(const std::vector<int> &true_mapping);

    // Viterbi inference
    void ComputeViterbi();
    RealT GetViterbiScore() const;
    std::vector<int> PredictPairingsViterbi() const;
    std::vector<RealT> ComputeViterbiFeatureCounts();

    // MEA inference
    void ComputeInside();
    RealT ComputeLogPartitionCoefficient() const;
    void ComputeOutside();
    std::vector<RealT> ComputeFeatureCountExpectations();
    void ComputePosterior();
    std::vector<int> PredictPairingsPosterior(const RealT gamma) const;
    RealT *GetPosterior(const RealT posterior_cutoff) const;
};

#include "InferenceEngine.ipp"

#endif
