/* 
 * In this version, input specifies the target data minus the txt in the filename:
 * 	--input=data/NewportAslin
 * and then we'll add in the data amounts from the directory (via the vector of strings data_amounts) * 
 * */
  
#include <set>
#include <string>
#include <vector>
#include <numeric> // for gcd
#include <filesystem>
#include <fstream>
#include <chrono>

#include "Data.h"

using S = std::string;
using StrSet = std::set<S>;

const std::string my_default_input = "data/English"; 

S alphabet="nvadtp";
size_t max_length = 128; // (more than 256 needed for count, a^2^n, a^n^2, etc -- see command line arg)
size_t max_setsize = 64; // throw error if we have more than this
size_t nfactors = 2; // how may factors do we run on? (defaultly)

static constexpr float alpha = 0.01; // probability of insert/delete errors (must be a float for the string function below)

size_t PREC_REC_N   = 25;  // number of top strings that we use to approximate precision and recall

const double MAX_TEMP = 1.20; 
unsigned long PRINT_STRINGS; // print at most this many strings for each hypothesis

//std::vector<S> data_amounts={"1", "2", "5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "50000", "100000"}; // how many data points do we run on?
std::vector<S> data_amounts={"100000"}; // 

size_t current_ntokens = 0; // how many tokens are there currently? Just useful to know

const std::string errorstring = "<err>";

#include "MyGrammar.h"
#include "MyHypothesis.h"

std::string prdata_path = "";
std::vector<MyHypothesis::datum_t> prdata_store;
MyHypothesis::data_t prdata = prdata_store; // used for computing precision and recall -- in case we want to use more strings?
S current_data = "";
bool long_output = false; // if true, we allow extra strings, recursions etc. on output
std::pair<double,double> mem_pr;

////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DO_NOT_INCLUDE_MAIN

#include "VirtualMachine/VirtualMachineState.h"
#include "VirtualMachine/VirtualMachinePool.h"
#include "TopN.h"
#include "ParallelTempering.h"
#include "Fleet.h" 

int main(int argc, char** argv){ 
	
	// Set this
	VirtualMachineControl::MIN_LP = -15;
	FleetArgs::nchains = 5; // default number is 5
	
	FleetArgs::input_path = my_default_input; // set this so it's not fleet's normal input default
	
	// default include to process a bunch of global variables: mcts_steps, mcc_steps, etc
	Fleet fleet("Formal language learner");
	fleet.add_option("-N,--nfactors",      nfactors, "How many factors do we run on?");
	fleet.add_option("-L,--maxlength",     max_length, "Max allowed string length");
	fleet.add_option("-A,--alphabet",  alphabet, "The alphabet of characters to use");
	fleet.add_option("-P,--prdata",  prdata_path, "What data do we use to compute precion/recall?");
	fleet.add_option("--prN",  PREC_REC_N, "How many data points to compute precision and recall?");	
	fleet.add_flag("-l,--long-output",  long_output, "Allow extra computation/recursion/strings when we output");
	fleet.initialize(argc, argv); 

	// create an output directory for structured results
	std::filesystem::path output_dir(FleetArgs::output_path);
	if(output_dir.empty()) output_dir = "output";
	std::filesystem::create_directories(output_dir);

	// save the exact invocation and basic configuration
	{
		std::ofstream config_out(output_dir / "config.txt");
		config_out << "command=";
		for(int i=0;i<argc;i++) {
			if(i) config_out << ' ';
			config_out << argv[i];
		}
		config_out << "\n";
		config_out << "output_dir=" << output_dir.string() << "\n";
		config_out << "input=" << FleetArgs::input_path << "\n";
		config_out << "prdata_path=" << prdata_path << "\n";
		config_out << "data_amounts=";
		for(size_t i=0;i<data_amounts.size();i++) {
			if(i) config_out << ",";
			config_out << data_amounts[i];
		}
		config_out << "\n";
		config_out << "alphabet=" << alphabet << "\n";
		config_out << "nfactors=" << nfactors << "\n";
		config_out << "maxlength=" << max_length << "\n";
		config_out << "long_output=" << (long_output ? "true" : "false") << "\n";
		config_out << "threads=" << FleetArgs::nthreads << "\n";
		config_out << "time=" << FleetArgs::timestring << "\n";
		config_out << "top=" << FleetArgs::ntop << "\n";
		config_out << "thin=" << FleetArgs::thin << "\n";
		config_out << "restart=" << FleetArgs::restart << "\n";
	}

	// since we are only storing the top, we can ignore repeats from MCMC 
	FleetArgs::MCMCYieldOnlyChanges = true;
	
	
	COUT "# Using alphabet=" << alphabet ENDL;

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// Set up the grammar using command line arguments
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	// each of the recursive calls we are allowed
	for(size_t i=0;i<nfactors;i++) {	
		grammar.add_terminal( str(i), (int)i, 1.0/nfactors);
	}
		
	for(const char c : alphabet) {
		grammar.add_terminal( Q(S(1,c)), c, CONSTANT_P/alphabet.length());
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// Load the data
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	// Input here is going to specify the PRdata path, minus the txt
	if(prdata_path == "") {	prdata_path = FleetArgs::input_path+".txt"; }
	
	load_data_file(prdata_store, prdata_path.c_str()); // put all the data in prdata_store
	prdata = prdata_store;
	for(auto d : prdata) { 	// Add a check for any data not in the alphabet
		check_alphabet(d.output, alphabet);
	}
	
	// We are going to build up the data
	std::vector<std::vector<MyHypothesis::datum_t>> datas_store;
	std::vector<MyHypothesis::data_t> datas; // load all the data
	for(size_t i=0;i<data_amounts.size();i++){
		datas_store.emplace_back();
		
		S data_path = FleetArgs::input_path + "-" + data_amounts[i] + ".txt";
		load_data_file(datas_store.back(), data_path.c_str());
		datas.emplace_back(datas_store.back());
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// Actually run
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
	// Build up an initial hypothesis with the right number of factors
	MyHypothesis h0;
	for(size_t i=0;i<nfactors;i++) { 
		h0[i] = InnerHypothesis::sample(); 
	}
 
	// where to store these hypotheses
	TopN<MyHypothesis> all; 
	
	ParallelTempering samp(h0, datas[0], FleetArgs::nchains, MAX_TEMP); 
//	ChainPool samp(h0, &datas[0], FleetArgs::nchains);
	
	// Set these up as the defaults as below
	VirtualMachineControl::MAX_STEPS  = 1024; // TODO: Change to MAX_RUN_PROGRAM or remove?
	VirtualMachineControl::MAX_OUTPUTS = 256; 
	VirtualMachineControl::MIN_LP = -15;
	PRINT_STRINGS = 512;	
	auto run_start = std::chrono::steady_clock::now();

	for(size_t di=0;di<datas.size() and !CTRL_C;di++) {
		auto& this_data = datas[di];
		
		samp.set_data(this_data, true);
		
		// compute the prevision and recall if we just memorize the data
		{
			double cz = 0; // find the normalizer for counts			
			for(auto& d: this_data) cz += d.count; 
			
			DiscreteDistribution<S> mem_d;
			for(auto& d: this_data) 
				mem_d.addmass(d.output, log(d.count)-log(cz)); 
			
			// set a global var to be used when printing hypotheses above
			mem_pr = get_precision_and_recall(mem_d, prdata, PREC_REC_N);
		}
		
		
		// set this global variable so we know
		current_ntokens = 0;
		for(auto& d : this_data) { UNUSED(d); current_ntokens++; }
//		for(auto& h : samp.run_thread(Control(FleetArgs::steps/datas.size(), FleetArgs::runtime/datas.size(), FleetArgs::nthreads, FleetArgs::restart))) {
		for(auto& h : samp.run(Control(FleetArgs::steps/datas.size(), FleetArgs::runtime/datas.size(), FleetArgs::nthreads, FleetArgs::restart))
						| printer(FleetArgs::print) | all) {
			UNUSED(h);
		}	

		// set up to print using a larger set if we were given this option
		if(long_output){
			VirtualMachineControl::MAX_STEPS  = 32000; 
			VirtualMachineControl::MAX_OUTPUTS = 16000;
			VirtualMachineControl::MIN_LP = -40;
			PRINT_STRINGS = 5000;
		}

		all.print(data_amounts[di]);
		
		{
			auto suffix = data_amounts[di];
			auto top_path = std::filesystem::path(FleetArgs::output_path) / ("top-hypotheses-" + suffix + ".txt");
			std::ofstream top_out(top_path);
			top_out << "distribution\tposterior\tprior\tlikelihood\tprecision\trecall\tstring\n";
			for(auto& h : all.sorted(false)) {
				auto o = h.call(EMPTY_STRING, errorstring);
				auto dist = o.string(PRINT_STRINGS);
				auto [prec, rec] = get_precision_and_recall(o, prdata, PREC_REC_N);
				top_out << dist << '\t' << h.posterior << '\t' << h.prior << '\t' << h.likelihood << '\t' << prec << '\t' << rec << '\t' << h.string() << '\n';
			}
		}
		
		// restore
		if(long_output) {
			VirtualMachineControl::MAX_STEPS  = 1024; 
			VirtualMachineControl::MAX_OUTPUTS = 256; 
			VirtualMachineControl::MIN_LP = -15;
			PRINT_STRINGS = 512;
		}	
		
		if(di+1 < datas.size()) {
			all = all.compute_posterior(datas[di+1]); // update for next time
		}
		
	}

	auto run_end = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration<double>(run_end - run_start).count();
	std::ofstream metrics_out(std::filesystem::path(FleetArgs::output_path) / "metrics.txt");
	metrics_out << "elapsed_seconds=" << elapsed_seconds << "\n";
	double samples_per_second = elapsed_seconds > 0 ? double(FleetStatistics::global_sample_count) / elapsed_seconds : 0.0;
	metrics_out << "samples_per_second=" << samples_per_second << "\n";
	metrics_out << "global_sample_count=" << FleetStatistics::global_sample_count << "\n";
	metrics_out << "depth_exceptions=" << FleetStatistics::depth_exceptions << "\n";
	metrics_out << "posterior_calls=" << FleetStatistics::posterior_calls << "\n";
	metrics_out << "vm_ops_per_second=" << ((FleetStatistics::vm_ops / 1000000.0) / (elapsed_seconds > 0 ? elapsed_seconds : 1.0)) << "\n";

}

#endif
