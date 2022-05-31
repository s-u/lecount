#ifndef DIGRAPH_H
#define DIGRAPH_H

#include "set.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <queue>
#include <unordered_map>
#include <vector>
#include <cstring>

#define eprintf(...) if (opt_verbose) fprintf (stderr, __VA_ARGS__)
#define eflush() fflush(stderr)

extern int opt_verbose;

#define ALGO_RECURSIVE 0
#define ALGO_BUCKET_ELIMINATION 1

#define CCS_NONE 0
#define CCS_DFS_SPLIT_ONCE 1
#define CCS_DFS_SPLIT_ALL 2
#define CCS_COVERS 3
#define CCS_AUTO 4

#define HUB_NONE 0
#define HUB_BEST 1
#define HUB_PERFECT 2

#define STATIC_NO 0
#define STATIC_YES 1

#define TRANSPOSE_NO 0
#define TRANSPOSE_YES 1
#define TRANSPOSE_AUTO 2
#define TRANSPOSE_AUTO2 3

#define SORT_NO 0
#define SORT_YES 1

#define ELIM_ORDER_DEFAULT 0
#define ELIM_ORDER_REVERSE 1
#define ELIM_ORDER_LEAST_DEGREE 2
#define ELIM_ORDER_TOPOLOGICAL 3



template <typename Set, typename N>
struct Item
{
	Set set;
	N ext;
	int next;
	Item() {}
};

static unsigned hash_primes[] = {
	53,
	97,
	193,
	389,
	769,
	1543,
	3079,
	6151,
	12289,
	24593,
	49157,
	98317,
	196613,
	393241,
	786433,
	1572869,
	3145739,
	6291469,
	12582917,
	25165843,
	50331653,
	100663319,
	201326611,
	402653189,
	805306457,
	1610612741
};


struct CacheOptions
{
	int initial_prime_index;
	int prime_index_increment;
	double max_load_factor;
	
	int initial_array_size;
	int array_resize_factor;
	
	CacheOptions()
	{
		initial_prime_index = 4;
		prime_index_increment = 3;
		max_load_factor = 0.5;
		
		initial_array_size = 1024;
		array_resize_factor = 4;
	}
};

template <typename Set, typename N>
struct Cache
{
	int *table;
	Item<Set, N> *array;
	
	unsigned table_size;
	unsigned array_size;
	unsigned hash_prime_index;
	
	unsigned used;
	
	CacheOptions options;
	
	Cache(CacheOptions options) : options(options)
	{
		eprintf("Allocating cache (%i, %i, %f, %i, %i)...\n",
			options.initial_prime_index,
			options.prime_index_increment,
			options.max_load_factor,
			options.initial_array_size,
			options.array_resize_factor
		);
		
		used = 0;
		
		hash_prime_index = options.initial_prime_index;
		table_size = hash_primes[hash_prime_index];
		array_size = options.initial_array_size;
		
		table = new int[table_size];
		array = new Item<Set, N>[array_size];
		
		for (unsigned i = 0; i < table_size; ++i) {
			table[i] = -1;
		}
	}
	
	~Cache()
	{
		delete [] table;
		delete [] array;
	}
	
	void rehash(unsigned new_size)
	{
		delete [] table;
		
		table_size = new_size;
		
		table = new int[table_size];
		
		for (unsigned i = 0; i < table_size; ++i) {
			table[i] = -1;
		}
		
		for (unsigned i = 0; i < used; ++i) {
			array[i].next = -1;
		}
		
		for (unsigned add = 0; add < used; ++add) {
			unsigned h1 = std::hash<Set>()(array[add].set) % table_size;
			
			int i = table[h1];
			int prev = -1;
			
			while (i != -1) {
				prev = i;
				i = array[i].next;
			}
			
			if (prev == -1) {
				table[h1] = add;
			} else {
				array[prev].next = add;
			}
		}
	}
	
	int allocate_new()
	{
		if (used == array_size) {
			int new_size = array_size * options.array_resize_factor;
			eprintf("  Reallocating to... %u\n", new_size);
			Item<Set, N> *new_array = new Item<Set, N>[new_size];
			std::copy(array, array + array_size, new_array);
			delete [] array;
			array = new_array;
			array_size = new_size;
		}
		
		if (used >= table_size * options.max_load_factor) {
			hash_prime_index += options.prime_index_increment;
			
			if (hash_prime_index > 25) hash_prime_index = 25;
			
			unsigned size = hash_primes[hash_prime_index];
			
			eprintf("  Rehashing to %u...\n", size);
			rehash(size);
		}
		
		return used++;
	}
	
	int count(Set &set)
	{
		std::size_t h1 = std::hash<Set>()(set) % table_size;
		
		int i = table[h1];
		
		while (i != -1) {
			if (array[i].set == set) return 1;
			i = array[i].next;
		}
		
		return 0;
	}
	
	N& operator [] (Set &set)
	{
		std::size_t h1 = std::hash<Set>()(set) % table_size;
		
		int i = table[h1];
		int prev = -1;
		
		while (i != -1) {
			if (array[i].set == set) return array[i].ext;
			prev = i;
			i = array[i].next;
		}
		
		int add = allocate_new();
		array[add].set = set;
		array[add].next = -1;
		
		if (prev == -1) {
			table[h1] = add;
		} else {
			array[prev].next = add;
		}
		
		return array[add].ext;
	}
};






struct LECOptions
{
	int algorithm;
	int transpose;
	int ccs;
	double ccs_auto_threshold;
	int print_subsets;
	int print_cache;
	int hub_splits;
	int static_partitions;
	int sort_topologically;
	int elimination_order;
	const char *subposet;
	const char *elim_order_file;
	CacheOptions cache;
	
	LECOptions()
	{
		algorithm = ALGO_RECURSIVE;
		transpose = TRANSPOSE_AUTO;
		ccs = CCS_AUTO;
		ccs_auto_threshold = 3.3;
		print_subsets = 0;
		print_cache = 0;
		hub_splits = HUB_NONE;
		static_partitions = STATIC_NO;
		sort_topologically = SORT_YES;
		elimination_order = ELIM_ORDER_LEAST_DEGREE;
		subposet = NULL;
		elim_order_file = NULL;
	}
};



struct Statistics
{
	long long unsigned int recursive_calls;
	long long unsigned int component_splits;
	long long unsigned int cache_retrievals;
	long long unsigned int evaluated_subgraphs;
	long long unsigned int connected_subgraphs;
	long long unsigned int trivial_cases;
	long long unsigned int minimal_branchings;
	long long unsigned int maximal_removals;
	long long unsigned int child_symmetries;
	long long unsigned int hub_splits;
	long long unsigned int static_partitions;
	long long unsigned int connectivity_checks;
	long long unsigned int acyclic_subgraphs;
	
	Statistics()
	{
		recursive_calls = 0;
		component_splits = 0;
		cache_retrievals = 0;
		evaluated_subgraphs = 0;
		connected_subgraphs = 0;
		trivial_cases = 0;
		minimal_branchings = 0;
		maximal_removals = 0;
		child_symmetries = 0;
		hub_splits = 0;
		static_partitions = 0;
		connectivity_checks = 0;
		acyclic_subgraphs = 0;
	}
};



template <typename N, typename Set>
struct RecInfo
{
	LECOptions options;
	
	N **binomials;
	
	Cache<Set, N> cache;
	
	unsigned *subset_elements;
	unsigned n_subset_elements;
	unsigned *n_subset_predecessors;
	Set *covers;
	unsigned cn;
	
	Set *downsets;
	Set *upsets;
	
	Statistics stats;
	
	int ccs_method;
	int hub_splits;
	
	RecInfo(unsigned n, N **binomials_, LECOptions options) :
		options(options),
		binomials(binomials_),
		cache(options.cache),
		subset_elements(new unsigned[n]),
		downsets(new Set[n]),
		upsets(new Set[n])
	{
	}
	
	~RecInfo()
	{
		delete [] subset_elements;
		delete [] downsets;
		delete [] upsets;
	}
	
	N binom(unsigned n, unsigned k) const
	{
		if (n-k < k) k = n-k;
		return binomials[n][k];
	}
};






struct LinearExtensionCounter
{
	bool *rel;
	unsigned *neighbors;
	unsigned *neighbors_n;
	unsigned *children;
	unsigned *children_n;
	unsigned *minimals;
	
	int induced_width;
	
	bool& pair(unsigned i, unsigned j)
	{
		return rel[i * n + j];
	}
	
	unsigned n;
	
	void flip(unsigned i, unsigned j)
	{
		pair(i, j) ^= 1;
	}
	
	void add(unsigned i, unsigned j)
	{
		pair(i, j) = 1;
	}
	
	void del(unsigned i, unsigned j)
	{
		pair(i, j) = 0;
	}
	
	bool has(unsigned i, unsigned j) const
	{
		return (*this)(i, j);
	}
	
	bool operator() (unsigned i, unsigned j) const
	{
		return rel[i * n + j];
	}
	
	void init(unsigned n)
	{
		rel = new bool[n * n];
		neighbors = new unsigned[n * n];
		neighbors_n = new unsigned[n];
		children = new unsigned[n * n];
		children_n = new unsigned[n];
		minimals = new unsigned[n];
	}
	
	void uninit()
	{
		delete [] rel;
		delete [] neighbors;
		delete [] neighbors_n;
		delete [] children;
		delete [] children_n;
		delete [] minimals;
	}
	
	LinearExtensionCounter(unsigned n) : n(n)
	{
		init(n);
		
		for (unsigned i = 0; i < n; i++) {
			for (unsigned j = 0; j < n; j++) {
				pair(i,j) = 0;
			}
		}
	}
	
	~LinearExtensionCounter()
	{
		uninit();
	}
	
	template <typename N, typename Set>
	void print_dot(const char *name, RecInfo<N, Set> &rec, Set X, int mode, int vertex) const
	{
		FILE *f = fopen(name, "w");
		print_dot(f, rec, X, mode, vertex);
		fclose(f);
	}
	
	void invert()
	{
		for (unsigned i = 0; i < n-1; i++) {
			for (unsigned j = i+1; j < n; j++) {
				if (has(i, j) != has(j, i)) {
					flip(i, j);
					flip(j, i);
				}
			}
		}
	}
	
	void tree_sort_visit(std::vector<unsigned> &sorted, bool *visited, unsigned i) const
	{
		if (visited[i]) return;
		visited[i] = true;
		
		for (unsigned j = 0; j < n; j++) {
			if (has(j, i) || has(i, j)) tree_sort_visit(sorted, visited, j);
		}
		
		sorted.push_back(i);
	}
	
	void find_tree_ordering(std::vector<unsigned> &sorted)
	{
		bool visited[n];
		
		for (unsigned i = 0; i < n; i++) {
			visited[i] = false;
		}
		
		for (unsigned i = 0; i < n; i++) {
			tree_sort_visit(sorted, visited, i);
		}
	}
	
	void find_least_degree_ordering(std::vector<unsigned> &sorted)
	{
		int v_neighbors[n];
		int remaining[n];
		for (unsigned i = 0; i < n; ++i) {
			v_neighbors[i] = 0;
			remaining[i] = i;
		}
		for (unsigned i = 0; i < n-1; ++i) {
			for (unsigned j = i+1; j < n; ++j) {
				if (has(i,j) || has(j,i)) {
					v_neighbors[i]++;
					v_neighbors[j]++;
				}
			}
		}
		
		for (unsigned k = 0; k < n; ++k) {
			int j = 0;
			for (unsigned i = 1; i < n-k; ++i) {
				if (v_neighbors[remaining[i]] < v_neighbors[remaining[j]]) j = i;
			}
			int v = remaining[j];
			sorted.push_back(v);
			remaining[j] = remaining[n-k-1];
			v_neighbors[v]--;
			for (unsigned i = 0; i < n; ++i) {
				if (has(v,i) || has(i,v)) v_neighbors[i]--;
			}
		}
	}
	
	void topological_sort_visit(std::vector<unsigned> &sorted, bool *visited, unsigned i) const
	{
		if (visited[i]) return;
		
		for (unsigned j = 0; j < n; j++) {
			if (has(j, i)) topological_sort_visit(sorted, visited, j);
		}
		
		visited[i] = true;
		sorted.push_back(i);
	}
	
	void find_topological_ordering(std::vector<unsigned> &sorted)
	{
		bool visited[n];
		
		for (unsigned i = 0; i < n; i++) {
			visited[i] = false;
		}
		
		for (unsigned i = 0; i < n; i++) {
			if (!is_maximal(i)) continue;
			topological_sort_visit(sorted, visited, i);
		}
	}
	
	void sort_topologically()
	{
		std::vector<unsigned> sorted;
		find_topological_ordering(sorted);
		
		bool *new_rel = new bool[n * n];
		
		for (unsigned i = 0; i < n; i++) {
			for (unsigned j = 0; j < n; j++) {
				new_rel[i * n + j] = has(sorted[i], sorted[j]);
			}
		}
		
		delete [] rel;
		rel = new_rel;
	}
	
	void longest_paths(bool *visited, int *lpath, unsigned u) const
	{
		if (visited[u]) return;
		
		for (unsigned v = 0; v < n; v++) {
			if (!has(u, v)) continue;
			
			longest_paths(visited, lpath, v);
			
			for (unsigned w = 0; w < n; w++) {
				if (lpath[v*n+w] == -1) continue;
				
				int len = lpath[v*n+w] + 1;
				if (len > lpath[u*n+w]) lpath[u*n+w] = len;
			}
		}
		
		visited[u] = true;
	}
	
	void longest_paths(int *lpath) const
	{
		bool visited[n];
		
		for (unsigned u = 0; u < n; u++) {
			visited[u] = false;
			for (unsigned v = 0; v < n; v++) {
				lpath[u*n+v] = (u == v) ? 0 : -1;
			}
		}
		
		for (unsigned u = 0; u < n; u++) {
			longest_paths(visited, lpath, u);
		}
	}
	
	void remove_transitive_arcs()
	{
		eprintf("  Computing transitive reduction...\n");
		
		int lpath[n*n];
		longest_paths(lpath);
		
		for (unsigned u = 0; u < n; u++) {
			for (unsigned v = 0; v < n; v++) {
				pair(u, v) = lpath[u*n+v] == 1;
			}
		}
	}
	
	void take_transitive_closure()
	{
		eprintf("  Computing transitive closure...\n");
		
		int lpath[n*n];
		longest_paths(lpath);
		
		for (unsigned u = 0; u < n; u++) {
			for (unsigned v = 0; v < n; v++) {
				pair(u, v) = lpath[u*n+v] >= 1;
			}
		}
	}
	
	void print() const
	{
		for (unsigned j = 0; j < n; j++) {
			for (unsigned i = 0; i < n; i++) {
				printf("%s", has(i, j) ? "#" : "-");
				if (i < n - 1) printf(" ");
			}
			printf("\n");
		}
	}
	
	bool is_maximal(unsigned u) const
	{
		for (unsigned v = 0; v < n; v++) {
			if (has(u, v)) return false;
		}
		
		return true;
	}
	
	unsigned get_maximal_elements(unsigned *maximal) const
	{
		unsigned k = 0;
		for (unsigned u = 0; u < n; u++) {
			if (is_maximal(u)) {
				maximal[k] = u;
				k++;
			}
		}
		return k;
	}
	
	template <typename Set>
	bool is_maximal(unsigned i, Set X) const
	{
		for (unsigned j = 0; j < n; j++) {
			if (!X[j]) continue;
			if (has(i, j)) return false;
		}
		
		return true;
	}
	
	template <typename Set>
	Set get_maximal_elements(Set X) const
	{
		Set maximal = Set::empty(n);
		for (unsigned i = 0; i < n; i++) {
			if (!X[i]) continue;
			if (is_maximal(i, X)) maximal.set(i);
		}
		return maximal;
	}
	
	template <typename Set>
	unsigned get_maximal_elements(unsigned *maximal, Set X) const
	{
		unsigned k = 0;
		for (unsigned i = 0; i < n; i++) {
			if (!X[i]) continue;
			if (is_maximal(i, X)) {
				maximal[k] = i;
				k++;
			}
		}
		return k;
	}
	
	template <typename Set>
	void undirected_DFS(Set &cmp, unsigned i, Set X) const
	{
		cmp.set(i);
		
		for (unsigned k = 0; k < n; k++) {
			if (!X[k]) continue;
			if (!has(i, k) && !has(k, i)) continue;
			if (cmp[k]) continue;
			undirected_DFS(cmp, k, X);
		}
	}
	
	template <typename Set>
	void undirected_DFS_nlist(Set &cmp, unsigned i, Set X) const
	{
		cmp.set(i);
		
		for (unsigned k = 0; k < neighbors_n[i]; k++) {
			int j = neighbors[i*n+k];
			if (!X[j]) continue;
			if (cmp[j]) continue;
			undirected_DFS_nlist(cmp, j, X);
		}
	}
	
	template <typename Set>
	Set find_undirected_component(unsigned i, Set X) const
	{
		Set cmp = Set::empty(n);
		undirected_DFS_nlist(cmp, i, X);
		return cmp;
	}
	
	template <typename Set>
	void child_DFS_nlist(Set &cmp, unsigned i) const
	{
		cmp.set(i);
		
		for (unsigned k = 0; k < children_n[i]; k++) {
			int j = children[i*n+k];
			if (cmp[j]) continue;
			child_DFS_nlist(cmp, j);
		}
	}
	
	template <typename Set>
	Set find_descendants(unsigned i) const
	{
		Set cmp = Set::empty(n);
		child_DFS_nlist(cmp, i);
		return cmp;
	}
	
	template <typename Set>
	void parent_DFS_nlist(Set &cmp, unsigned i) const
	{
		cmp.set(i);
		
		for (unsigned u = 0; u < n; u++) {
			if (cmp[u]) continue;
			if (has(u, i)) parent_DFS_nlist(cmp, u);
		}
	}
	
	template <typename Set>
	Set find_ancestors(unsigned i) const
	{
		Set cmp = Set::empty(n);
		parent_DFS_nlist(cmp, i);
		return cmp;
	}
	
	template <typename Set>
	Set get_parents(unsigned i, Set X) const
	{
		Set ps = Set::empty(n);
		for (unsigned k = 0; k < n; k++) {
			if (!X[k]) continue;
			if (has(k, i)) ps.set(k);
		}
		return ps;
	}
	
	template <typename Set>
	Set get_children(unsigned i, Set X) const
	{
		Set cs = Set::empty(n);
		for (unsigned k = 0; k < n; k++) {
			if (!X[k]) continue;
			if (has(i, k)) cs.set(k);
		}
		return cs;
	}
	
	template <typename Set>
	Set get_parents(unsigned u) const
	{
		Set parents = Set::empty(n);
		for (unsigned v = 0; v < n; v++) {
			if (has(v, u)) parents.set(v);
		}
		return parents;
	}
	
	void compute_neighbor_lists()
	{
		for (unsigned i = 0; i < n; i++) {
			neighbors_n[i] = 0;
			children_n[i] = 0;
		}
		
		for (unsigned i = 0; i < n; i++) {
			for (unsigned j = i+1; j < n; j++) {
				if (has(i,j) || has(j,i)) {
					neighbors[i*n+neighbors_n[i]] = j;
					neighbors[j*n+neighbors_n[j]] = i;
					neighbors_n[i]++;
					neighbors_n[j]++;
				}
			}
		}
		
		for (unsigned i = 0; i < n; i++) {
			for (unsigned j = 0; j < n; j++) {
				if (has(i,j)) {
					children[i*n+children_n[i]] = j;
					children_n[i]++;
				}
			}
		}
	}
	
	void leaf_distance_visit(int *level, unsigned i) const
	{
		if (level[i] >= 0) return;
		
		int max_child_level = -1;
		for (unsigned j = 0; j < n; j++) {
			if (!has(i, j)) continue;
			leaf_distance_visit(level, j);
			if (level[j] > max_child_level) max_child_level = level[j];
		}
		
		level[i] = max_child_level + 1;
	}
	
	void root_distance_visit(int *level, unsigned i) const
	{
		if (level[i] >= 0) return;
		
		int max_parent_level = -1;
		for (unsigned j = 0; j < n; j++) {
			if (!has(j, i)) continue;
			root_distance_visit(level, j);
			if (level[j] > max_parent_level) max_parent_level = level[j];
		}
		
		level[i] = max_parent_level + 1;
	}
	
	double average_degree() const
	{
		int edges = 0;
		for (unsigned i = 0; i < n; i++) {
			for (unsigned j = 0; j < n; j++) {
				if (has(i, j)) edges++;
			}
		}
		
		return 2.0 * edges / n;
	}
	
	bool transpose_heuristic() const
	{
		if (n < 3) return false;
		
		int llevel[n];
		unsigned lcount[n];
		int rlevel[n];
		unsigned rcount[n];
		
		for (unsigned i = 0; i < n; i++) {
			llevel[i] = -1;
			lcount[i] = 0;
			rlevel[i] = -1;
			rcount[i] = 0;
		}
		
		for (unsigned i = 0; i < n; i++) {
			leaf_distance_visit(llevel, i);
			lcount[llevel[i]]++;
			root_distance_visit(rlevel, i);
			rcount[rlevel[i]]++;
		}
		
		unsigned k = n;
		while (lcount[k-1] == 0) k--;
		
		eprintf("    Minimal: %i   Maximal: %i\n", rcount[0], lcount[0]);
		bool t = rcount[0] >= lcount[0];
		
		return t;
	}
	
	template <typename Set>
	void get_undirected_component(unsigned i, Set &subset, unsigned **next) const
	{
		subset.flip(i);
		**next = i;
		(*next)++;
		
		for (unsigned k = 0; k < neighbors_n[i]; ++k) {
			int j = neighbors[i*n+k];
			if (!subset[j]) continue;
			get_undirected_component(j, subset, next);
		}
	}
	
	template <typename Set>
	bool is_minimal(unsigned i, Set X) const
	{
		for (unsigned j = 0; j < n; j++) {
			if (!X[j]) continue;
			if (has(j, i)) return false;
		}
		
		return true;
	}
	
	template <typename Set>
	Set get_minimal_elements(Set X) const
	{
		Set minimal = Set::empty(n);
		for (unsigned i = 0; i < n; i++) {
			if (!X[i]) continue;
			if (is_minimal(i, X)) minimal.set(i);
		}
		return minimal;
	}
	
	template <typename Set>
	double estimate_hardness_connected(Set subset) const
	{
		Set minimal = get_minimal_elements(subset);
		subset = subset ^ minimal;
		
		return pow(2, minimal.cardinality(n)) + estimate_hardness(subset);
	}
	
	template <typename Set>
	double estimate_hardness(Set subset) const
	{
		unsigned subset_elements[n];
		unsigned n_subset_elements = 0;
		
		for (unsigned i = 0; i < n; ++i) {
			if (subset.has(i)) subset_elements[n_subset_elements++] = i;
		}
		
		double estimate = 0.0;
		
		Set remaining = subset;
		
		unsigned cmp_elements[n_subset_elements];
		for (unsigned i = 0; i < n_subset_elements; ++i) {
			unsigned u = subset_elements[i];
			if (!remaining.has(u)) continue;
			
			unsigned *next = cmp_elements;
			get_undirected_component(u, remaining, &next);
			unsigned n_cmp_elements = next - cmp_elements;
			
			Set cmp = Set::empty(n);
			for (unsigned j = 0; j < n_cmp_elements; ++j) {
				cmp.set(cmp_elements[j]);
			}
			
			estimate += estimate_hardness_connected(cmp);
		}
		
		return estimate;
	}
	
	template <typename Set>
	bool transpose_heuristic2()
	{
		compute_neighbor_lists();
		double estimate_no = estimate_hardness(Set::complete(n));
		
		invert();
		compute_neighbor_lists();
		double estimate_yes = estimate_hardness(Set::complete(n));
		
		eprintf("    No: %.2f   Yes: %.2f\n", estimate_no, estimate_yes);
		
		invert();
		return estimate_no > estimate_yes;
	}
	
	template <typename N, typename Set>
	int find_central_vertex(RecInfo<N, Set> &rec, Set subset) const
	{
		int d = -1;
		int d_comparable = -1;
		for (unsigned i = 0; i < rec.n_subset_elements; i++) {
			unsigned u = rec.subset_elements[i];
			Set downset = rec.downsets[u] & subset;
			Set upset = rec.upsets[u] & subset;
			
			Set comparable = downset | upset;
			int u_comparable = comparable.cardinality(n);
			
			if (u_comparable > d_comparable) {
				d = u;
				d_comparable = u_comparable;
			}
		}
		
		return d;
	}
	
	template <typename N, typename Set>
	N count_linex_admissible_partitions(RecInfo<N, Set> &rec, Set subset)
	{
		int d = find_central_vertex(rec, subset);
		assert(d != -1);
		
		Set A_base = rec.downsets[d] & subset;
		Set B_base = rec.upsets[d] & subset;
		
		Set C = subset ^ d ^ A_base ^ B_base;
		
		rec.stats.hub_splits++;
		
		N sum_all_partitions = 0;
		
		std::unordered_map<Set, unsigned int *> datas;
		std::queue<Set> ideals;
		
		Set empty = Set::empty(n);
		ideals.push(empty);
		datas[empty] = new unsigned int[n];
		
		for (unsigned v = 0; v < n; ++v) {
			if (!C.has(v)) continue;
			datas[empty][v] = 0;
			for (unsigned u = 0; u < n; ++u) {
				if (!C.has(u)) continue;
				if (has(u, v)) ++datas[empty][v];
			}
		}
		
		while (!ideals.empty()) {
			Set X = ideals.front(); ideals.pop();
			
			Set A = A_base | X;
			Set B = B_base | (C ^ X);
			
			unsigned *store_subset_elements = rec.subset_elements;
			unsigned store_n_subset_elements = rec.n_subset_elements;
			unsigned *store_n_subset_predecessors = rec.n_subset_predecessors;
			
			unsigned A_elements[n];
			unsigned A_n_elements = 0;
			for (unsigned u = 0; u < n; u++) {
				if (A.has(u)) A_elements[A_n_elements++] = u;
			}
			unsigned B_elements[n];
			unsigned B_n_elements = 0;
			for (unsigned u = 0; u < n; u++) {
				if (B.has(u)) B_elements[B_n_elements++] = u;
			}
			
			rec.subset_elements = A_elements;
			rec.n_subset_elements = A_n_elements;
			rec.n_subset_predecessors = new unsigned[n];
			for (unsigned u = 0; u < n; u++) {
				rec.n_subset_predecessors[u] = get_parents(u, A).cardinality(n);
			}
			
			N part1 = count_linex_recursive(rec, A);
			
			delete [] rec.n_subset_predecessors;
			
			rec.subset_elements = B_elements;
			rec.n_subset_elements = B_n_elements;
			rec.n_subset_predecessors = new unsigned[n];
			for (unsigned u = 0; u < n; u++) {
				rec.n_subset_predecessors[u] = get_parents(u, B).cardinality(n);
			}
			
			N part2 = count_linex_recursive(rec, B);
			
			delete [] rec.n_subset_predecessors;
			
			rec.subset_elements = store_subset_elements;
			rec.n_subset_elements = store_n_subset_elements;
			rec.n_subset_predecessors = store_n_subset_predecessors;
			
			sum_all_partitions += part1 * part2;
			
			
			
			
			unsigned int *xdata = datas[X];
			
			for (unsigned u = 0; u < n; ++u) {
				if (!C.has(u)) continue;
				if (xdata[u] > 0) continue;
				Set Y = X | u;
				if (Y == X) continue;
				
				if (datas.count(Y) == 0) {
					unsigned int *ydata;
					ydata = new unsigned int[n];
					datas[Y] = ydata;
					ideals.push(Y);
					
					for (unsigned v = 0; v < n; ++v) {
						if (!C.has(v)) continue;
						if (!Y[v]) ydata[v] = xdata[v] - has(u, v);
					}
				}
			}
			
			delete [] xdata;
			datas.erase(X);
		}
		
		return rec.cache[subset] = sum_all_partitions;
	}
	
	
	
	template <typename N, typename Set>
	int find_hub_vertex(RecInfo<N, Set> &rec, Set subset)
	{
		for (unsigned i = 0; i < rec.n_subset_elements; i++) {
			unsigned u = rec.subset_elements[i];
			
			Set A = rec.downsets[u] & subset;
			Set B = rec.upsets[u] & subset;
			
			if ((subset ^ u ^ A ^ B).is_empty()) return u;
		}
		
		return -1;
	}
	
	template <typename N, typename Set>
	N count_linex_hub_split(RecInfo<N, Set> &rec, Set subset)
	{
		int d = find_hub_vertex(rec, subset);
		if (d == -1) return -1;

		Set A = rec.downsets[d] & subset;
		Set B = rec.upsets[d] & subset;
		
		rec.stats.hub_splits++;
		
		unsigned *store_subset_elements = rec.subset_elements;
		unsigned store_n_subset_elements = rec.n_subset_elements;
		unsigned *store_n_subset_predecessors = rec.n_subset_predecessors;
		
		
		unsigned A_elements[n];
		unsigned A_n_elements = 0;
		for (unsigned u = 0; u < n; u++) {
			if (A.has(u)) A_elements[A_n_elements++] = u;
		}
		rec.subset_elements = A_elements;
		rec.n_subset_elements = A_n_elements;
		rec.n_subset_predecessors = new unsigned[n];
		for (unsigned u = 0; u < n; u++) {
			rec.n_subset_predecessors[u] = get_parents(u, A).cardinality(n);
		}
		N part1 = count_linex_recursive(rec, A);
		delete [] rec.n_subset_predecessors;
		
		
		unsigned B_elements[n];
		unsigned B_n_elements = 0;
		for (unsigned u = 0; u < n; u++) {
			if (B.has(u)) B_elements[B_n_elements++] = u;
		}
		rec.subset_elements = B_elements;
		rec.n_subset_elements = B_n_elements;
		rec.n_subset_predecessors = new unsigned[n];
		for (unsigned u = 0; u < n; u++) {
			rec.n_subset_predecessors[u] = get_parents(u, B).cardinality(n);
		}
		N part2 = count_linex_recursive(rec, B);
		delete [] rec.n_subset_predecessors;
		
		
		rec.subset_elements = store_subset_elements;
		rec.n_subset_elements = store_n_subset_elements;
		rec.n_subset_predecessors = store_n_subset_predecessors;
		
		return rec.cache[subset] = part1 * part2;
	}
	
	template <typename N, typename Set>
	N count_linex_static_sets(RecInfo<N, Set> &rec, Set subset)
	{
		int lower[n];
		int upper[n];
		
		for (int i = 0; i < rec.n_subset_elements; ++i) {
			int u = rec.subset_elements[i];
			lower[u] = (rec.downsets[u] & subset).cardinality(n);
			upper[u] = rec.n_subset_elements - (rec.upsets[u] & subset).cardinality(n);
		}
		
		int sorted[rec.n_subset_elements];
		
		for (int i = 0; i < rec.n_subset_elements; ++i) {
			int u = rec.subset_elements[i];
			int j = 0;
			while (j < i && ((lower[u] > lower[sorted[j]]) || (lower[u] == lower[sorted[j]] && upper[u] < upper[sorted[j]]))) ++j;
			for (int k = i; k > j; --k) sorted[k] = sorted[k-1];
			sorted[j] = rec.subset_elements[i];
		}
		
		N ext_n = 1;
		
		unsigned static_set_elements[rec.n_subset_elements];
		unsigned n_static_set_elements = 0;
		Set static_set = Set::empty(n);
		int min_lower = rec.n_subset_elements;
		int max_upper = 0;
		for (int i = 0; i < rec.n_subset_elements; ++i) {
			int u = sorted[i];
			
			static_set ^= u;
			static_set_elements[n_static_set_elements] = u;
			++n_static_set_elements;
			
			if (lower[u] < min_lower) min_lower = lower[u];
			if (upper[u] > max_upper) max_upper = upper[u];
			
			int span = max_upper - min_lower;
			if (span == rec.n_subset_elements) return -1;
			
			if (span == n_static_set_elements) {
				unsigned *store_subset_elements = rec.subset_elements;
				unsigned store_n_subset_elements = rec.n_subset_elements;
				unsigned *store_n_subset_predecessors = rec.n_subset_predecessors;
				
				rec.subset_elements = static_set_elements;
				rec.n_subset_elements = n_static_set_elements;
				rec.n_subset_predecessors = new unsigned[n];
				for (unsigned v = 0; v < n; v++) {
					rec.n_subset_predecessors[v] = get_parents(v, static_set).cardinality(n);
				}
				
				ext_n *= count_linex_recursive(rec, static_set);
				
				delete [] rec.n_subset_predecessors;
				
				rec.subset_elements = store_subset_elements;
				rec.n_subset_elements = store_n_subset_elements;
				rec.n_subset_predecessors = store_n_subset_predecessors;
				
				n_static_set_elements = 0;
				static_set = Set::empty(n);
				min_lower = rec.n_subset_elements;
				max_upper = 0;
			}
		}
		
		rec.stats.static_partitions++;
		
		return rec.cache[subset] = ext_n;
	}
	
	template <typename N, typename Set>
	N count_linex_connected_components_dfs_split_all(RecInfo<N, Set> &rec, Set subset)
	{
		Set remaining = subset;
		
		N ext_n = 1;
		unsigned n_cumul = 0;
		
		unsigned *store_subset_elements = rec.subset_elements;
		unsigned store_n_subset_elements = rec.n_subset_elements;
		
		unsigned cmp_elements[rec.n_subset_elements];
		for (unsigned i = 0; i < store_n_subset_elements; ++i) {
			unsigned u = store_subset_elements[i];
			if (!remaining.has(u)) continue;
			
			unsigned *next = cmp_elements;
			get_undirected_component(u, remaining, &next);
			unsigned n_cmp_elements = next - cmp_elements;
			
			if (i == 0) {
				if (n_cmp_elements == store_n_subset_elements) return -1;
				rec.stats.component_splits++;
			}
			
			Set cmp = Set::empty(n);
			for (unsigned j = 0; j < n_cmp_elements; ++j) {
				cmp.set(cmp_elements[j]);
			}
			
			rec.subset_elements = cmp_elements;
			rec.n_subset_elements = n_cmp_elements;
			
			N cmp_ext_n = count_linex_recursive(rec, cmp);
			
			n_cumul += n_cmp_elements;
			ext_n *= cmp_ext_n * rec.binom(n_cumul, n_cmp_elements);
		}
		
		rec.subset_elements = store_subset_elements;
		rec.n_subset_elements = store_n_subset_elements;
		
		return rec.cache[subset] = ext_n;
	}
	
	template <typename N, typename Set>
	N count_linex_connected_components_dfs_split_once(RecInfo<N, Set> &rec, Set subset)
	{
		Set cmp1 = find_undirected_component(rec.subset_elements[0], subset);
		if (cmp1 == subset) return -1;
		
		rec.stats.component_splits++;
		
		unsigned *elements = rec.subset_elements;
		unsigned n_elements = rec.n_subset_elements;
		
		Set cmp2 = Set::empty(n);
		unsigned elements1[n_elements];
		unsigned elements2[n_elements];
		unsigned n_elements1 = 0;
		unsigned n_elements2 = 0;
		
		for (unsigned i = 0; i < n_elements; i++) {
			if (cmp1[elements[i]]) {
				elements1[n_elements1++] = elements[i];
			} else {
				elements2[n_elements2++] = elements[i];
				cmp2.set(elements[i]);
			}
		}
		
		rec.subset_elements = elements1;
		rec.n_subset_elements = n_elements1;
		N cmp1_ext_n = count_linex_recursive(rec, cmp1, true);
		
		rec.subset_elements = elements2;
		rec.n_subset_elements = n_elements2;
		N cmp2_ext_n = count_linex_recursive(rec, cmp2);
		
		rec.subset_elements = elements;
		rec.n_subset_elements = n_elements;
		
		N bin = rec.binom(n_elements1 + n_elements2, n_elements2);
		
		return rec.cache[subset] = cmp1_ext_n * cmp2_ext_n * bin;
	}
	
	
	template <typename N, typename Set>
	N count_linex_connected_components_covers(RecInfo<N, Set> &rec, Set subset)
	{
		unsigned n_elements = rec.n_subset_elements;
		
		Set combined[rec.cn];
		unsigned k = 0;
		for (unsigned i = 0; i < rec.cn; i++) {
			combined[k] = rec.covers[i];
			combined[k] &= subset;
			if (!combined[k].is_empty()) {
				k++;
			}
		}
		
		for (unsigned i = 0; i < k; i++) {
			bool changes;
			do {
				changes = false;
				for (unsigned j = i+1; j < k; j++) {
					Set cut = combined[i];
					cut &= combined[j];
					if (cut.is_empty()) continue;
					combined[i] |= combined[j];
					k--;
					combined[j] = combined[k];
					j--;
					changes = true;
				}
			} while (changes);
		}
		
		if (k == 1) return -1;
		
		rec.stats.component_splits++;
		
		unsigned elements[n_elements];
		for (unsigned i = 0; i < n_elements; i++) elements[i] = rec.subset_elements[i];
		
		N ext_n = 1;
		unsigned rem_n_elements = n_elements;
		for (unsigned c = 0; c < k; c++) {
			unsigned cmp_n_elements = 0;
			Set &cmp = combined[c];
			for (unsigned i = 0; i < n_elements; i++) {
				if (cmp[elements[i]]) rec.subset_elements[cmp_n_elements++] = elements[i];
			}
			
			rec.n_subset_elements = cmp_n_elements;
			N bin = rec.binom(rem_n_elements, cmp_n_elements);
			
			N cmp_ext_n = count_linex_recursive(rec, cmp, true);
			
			ext_n *= cmp_ext_n * bin;
			rem_n_elements -= cmp_n_elements;
		}
		
		for (unsigned i = 0; i < n_elements; i++) rec.subset_elements[i] = elements[i];
		rec.n_subset_elements = n_elements;
		
		return rec.cache[subset] = ext_n;
	}
	
	
	template <typename N, typename Set>
	N count_linex_recursive_branch(RecInfo<N, Set> &rec, Set subset, unsigned v)
	{
		subset ^= v;
		
		if (rec.cache.count(subset) != 0) {
			rec.stats.recursive_calls++;
			rec.stats.cache_retrievals++;
			return rec.cache[subset];
		}
		
		for (unsigned i = 0; i < rec.n_subset_elements; i++) {
			unsigned u = rec.subset_elements[i];
			if (has(v, u)) rec.n_subset_predecessors[u]--;
		}
		
		N ext_n = count_linex_recursive(rec, subset);
		
		for (unsigned i = 0; i < rec.n_subset_elements; i++) {
			unsigned u = rec.subset_elements[i];
			if (has(v, u)) rec.n_subset_predecessors[u]++;
		}
		
		return ext_n;
	}
	
	template <typename N, typename Set>
	bool is_minimal(RecInfo<N, Set> &rec, unsigned u)
	{
		for (unsigned i = 0; i < rec.n_subset_elements; ++i) {
			unsigned v = rec.subset_elements[i];
			if (has(v, u)) return false;
		}
		
		return true;
	}
	
	template <typename N, typename Set>
	bool is_maximal(RecInfo<N, Set> &rec, unsigned u)
	{
		for (unsigned i = 0; i < rec.n_subset_elements; ++i) {
			unsigned v = rec.subset_elements[i];
			if (has(u, v)) return false;
		}
		
		return true;
	}
	
	template <typename N, typename Set>
	N count_linex_branching(RecInfo<N, Set> &rec, Set subset)
	{
		unsigned n_elements = rec.n_subset_elements;
		
		rec.n_subset_elements = n_elements - 1;
		
		rec.stats.minimal_branchings++;
		N sum = 0;
		
		for (unsigned i = 0; i < n_elements; i++) {
			unsigned v = rec.subset_elements[i];
			if (rec.n_subset_predecessors[v] > 0) continue;
			
			rec.subset_elements[i] = rec.subset_elements[rec.n_subset_elements];
			sum += count_linex_recursive_branch(rec, subset, v);
			rec.subset_elements[i] = v;
		}
		
		rec.n_subset_elements = n_elements;
		
		return rec.cache[subset] = sum;
	}
	
	template <typename Set>
	bool cycle_visit(int *visited, unsigned i, int p, Set X)
	{
		if (visited[i] == 2) return false;
		if (visited[i] == 1) return true;
		visited[i] = 1;
		
		for (unsigned k = 0; k < neighbors_n[i]; ++k) {
			int j = neighbors[i*n+k];
			if (!X[j]) continue;
			if (j == p) continue;
			if (cycle_visit(visited, j, i, X)) return true;
		}
		
		visited[i] = 2;
		return false;
	}
	
	template <typename Set>
	bool has_cycles(Set X)
	{
		int visited[n];
		for (unsigned k = 0; k < n; k++) {
			visited[k] = 0;
		}
		
		for (unsigned k = 0; k < n; k++) {
			if (cycle_visit(visited, k, -1, X)) return true;
		}
		
		return false;
	}
	
	template <typename N, typename Set>
	N count_linex_recursive(RecInfo<N, Set> &rec, Set subset, bool known_to_be_connected=false)
	{
		if (rec.options.print_subsets) subset.println(n);
		
		rec.stats.recursive_calls++;
		
		if (rec.cache.count(subset) != 0) {
			rec.stats.cache_retrievals++;
			return rec.cache[subset];
		}
		
		rec.stats.evaluated_subgraphs++;
		
		if (rec.n_subset_elements <= 1) {
			rec.stats.trivial_cases++;
			return rec.cache[subset] = 1;
		}
		
		if (rec.ccs_method > 0 && !known_to_be_connected) {
			N n_ext_cc;
			rec.stats.connectivity_checks++;
			if (rec.ccs_method == CCS_COVERS) {
				n_ext_cc = count_linex_connected_components_covers(rec, subset);
			} else if (rec.ccs_method == CCS_DFS_SPLIT_ONCE) {
				n_ext_cc = count_linex_connected_components_dfs_split_once(rec, subset);
			} else {
				n_ext_cc = count_linex_connected_components_dfs_split_all(rec, subset);
			}
			if (n_ext_cc != -1) return n_ext_cc;
		}
		
		rec.stats.connected_subgraphs++;
		
		if (rec.options.static_partitions) {
			N ext = count_linex_static_sets(rec, subset);
			if (ext != -1) return ext;
		}
		
		if (rec.hub_splits != HUB_NONE) {
			N ext;
			if (rec.hub_splits == HUB_BEST) {
				ext = count_linex_admissible_partitions(rec, subset);
			} else {
				ext = count_linex_hub_split(rec, subset);
			}
			if (ext != -1) return ext;
		}
		
		return count_linex_branching(rec, subset);
	}
	
	template <typename N, typename Set>
	void print_cache(Cache<Set, N> &cache)
	{
		for (unsigned i = 0; i < cache.used; ++i) {
			Item<Set, N> &item = cache.array[i];
			item.set.print(n);
			printf(" ");
			std::cout << item.ext << std::endl;
		}
	}
	
	template <typename Set>
	Set parse_set(const char *s)
	{
		Set set = Set::empty(n);
		int i = 0;
		while (*s != '\0' && i < n) {
			if (*s == '1') set ^= i;
			i++;
			s++;
		}
		return set;
	}
	
	template <typename Set>
	bool decide_transpose(LECOptions &options)
	{
		if (options.transpose == TRANSPOSE_NO) return false;
		if (options.transpose == TRANSPOSE_YES) return true;
		
		bool heuristic;
		
		if (options.transpose == TRANSPOSE_AUTO) {
			eprintf("  Running simple transpose heuristic...\n");
			heuristic = transpose_heuristic();
		} else {
			eprintf("  Running advanced transpose heuristic...\n");
			heuristic = transpose_heuristic2<Set>();
		}
		
		if (heuristic) {
			eprintf("    Transpose predicted easier.\n");
			return true;
		}
		
		eprintf("    Transpose predicted harder.\n");
		return false;
	}
	
	template <typename N, typename Set>
	N count_linex_recursive(N **binomials, LECOptions &options)
	{
		eprintf("Preprocessing...\n");
		
		double avg_deg = average_degree();
		
		remove_transitive_arcs();
		
		if (options.algorithm == ALGO_BUCKET_ELIMINATION) {
			return count_extensions(binomials, options);
		}
		
		if (options.sort_topologically) {
			eprintf("  Sorting topologically...\n");
			sort_topologically();
		}
		
		bool transpose = decide_transpose<Set>(options);
		
		if (transpose) {
			eprintf("  Transposing...\n");
			invert();
		}
		
		Set subset = Set::complete(n);
		if (options.subposet) {
			subset = parse_set<Set>(options.subposet);
		}
		
		eprintf("  Building successor list...\n");
		unsigned n_predecessors[n];
		for (unsigned v = 0; v < n; v++) {
			n_predecessors[v] = get_parents(v, subset).cardinality(n);
		}
		compute_neighbor_lists();
		
		RecInfo<N, Set> *rec_real = new RecInfo<N, Set>(n, binomials, options);
		RecInfo<N, Set> &rec = *rec_real;
		rec.n_subset_predecessors = n_predecessors;
		rec.n_subset_elements = 0;
		for (unsigned i = 0; i < n; i++) {
			if (subset.has(i)) rec.subset_elements[rec.n_subset_elements++] = i;
		}
		
		if (options.ccs == CCS_AUTO) {
			eprintf("  Choosing CCS method by average degree: %f\n", avg_deg);
			
			if (avg_deg < options.ccs_auto_threshold) {
				rec.ccs_method = CCS_DFS_SPLIT_ONCE;
			} else {
				rec.ccs_method = CCS_COVERS;
			}
		} else {
			rec.ccs_method = options.ccs;
		}
		
		eprintf("CCS method: ");
		if (rec.ccs_method == CCS_NONE) {
			eprintf("None\n");
		} else if (rec.ccs_method == CCS_COVERS) {
			eprintf("Maximal covers\n");
		} else if (rec.ccs_method == CCS_DFS_SPLIT_ONCE) {
			eprintf("DFS split once\n");
		} else if (rec.ccs_method == CCS_DFS_SPLIT_ALL) {
			eprintf("DFS split all\n");
		} else {
			assert(0);
		}
		
		rec.hub_splits = options.hub_splits;
		
		if (rec.hub_splits != HUB_NONE && rec.ccs_method == CCS_COVERS) {
			rec.ccs_method = CCS_DFS_SPLIT_ONCE;
			eprintf("  *** Forcing --ccs=dfs, since using partitioning\n");
			exit(1);
		}
		
		Set covers[n];
		
		if (rec.ccs_method == CCS_COVERS) {
			eprintf("  Finding maximal elements...\n");
			
			unsigned maximals[n];
			
			rec.cn = get_maximal_elements(maximals);
			
			eprintf("    Maximal elements: %i\n", rec.cn);
			eprintf("  Building covers...\n");
			
			rec.covers = covers;
			for (unsigned i = 0; i < rec.cn; i++) {
				rec.covers[i] = find_ancestors<Set>(maximals[i]);
			}
		}
		
		if (rec.hub_splits || rec.options.static_partitions) {
			eprintf("  Building downsets and upsets...\n");
			
			for (unsigned u = 0; u < n; u++) {
				rec.downsets[u] = find_ancestors<Set>(u) ^ u;
				rec.upsets[u] = find_descendants<Set>(u) ^ u;
			}
		}
		
		int opt_preprocess_only = 0;
		
		N n_ext = -1;
		
		if (!opt_preprocess_only) {
			eprintf("Solving...\n");
			
			n_ext = count_linex_recursive(rec, subset);
			
			eprintf("Solved. Deallocating...\n");
			
			long long unsigned n_connected = rec.stats.minimal_branchings
			                               + rec.stats.maximal_removals
			                               + rec.stats.hub_splits
			                               + rec.stats.static_partitions
			                               + rec.stats.child_symmetries;
			
			long long unsigned n_evaluations = rec.stats.trivial_cases
			                                 + rec.stats.component_splits
			                                 + rec.stats.connected_subgraphs;
			
			long long unsigned n_recursive_calls = rec.stats.cache_retrievals
			                                     + rec.stats.evaluated_subgraphs;
			
			assert(n_connected == rec.stats.connected_subgraphs);
			assert(n_evaluations == rec.stats.evaluated_subgraphs);
			assert(n_recursive_calls == rec.stats.recursive_calls);
			
			eprintf("Recursive calls:         %12llu\n", rec.stats.recursive_calls);
			eprintf("- Cache retrievals:      %12llu\n", rec.stats.cache_retrievals);
			eprintf("- Evaluations:           %12llu\n", rec.stats.evaluated_subgraphs);
			eprintf("  - Trivial:             %12llu\n", rec.stats.trivial_cases);
			eprintf("  - Disconnected:        %12llu\n", rec.stats.component_splits);
			eprintf("  - Connected:           %12llu\n", rec.stats.connected_subgraphs);
			eprintf("    - Min. branchings:   %12llu\n", rec.stats.minimal_branchings);
			eprintf("    - Max. removals:     %12llu\n", rec.stats.maximal_removals);
			eprintf("    - Hub splits:        %12llu\n", rec.stats.hub_splits);
			eprintf("    - Static partitions: %12llu\n", rec.stats.static_partitions);
			eprintf("    - Child symmetries:  %12llu\n", rec.stats.child_symmetries);
			eprintf("Connectivity checks:     %12llu\n", rec.stats.connectivity_checks);
			eprintf("Acyclic subgraphs:       %12llu\n", rec.stats.acyclic_subgraphs);
		}
		
		if (options.print_cache) print_cache(rec.cache);
		
		delete rec_real;
		
		return n_ext;
	}
	
	template <typename N>
	N count_linex(N **binomials, LECOptions &options)
	{
		N extn;
		if (n <= 32) {
			extn = count_linex_recursive<N, set32>(binomials, options);
		} else if (n <= 64) {
			extn = count_linex_recursive<N, set64>(binomials, options);
		} else if (n <= 128) {
			extn = count_linex_recursive<N, fixed_set<2>>(binomials, options);
		} else if (n <= 256) {
			extn = count_linex_recursive<N, fixed_set<4>>(binomials, options);
		} else if (n <= 512) {
			extn = count_linex_recursive<N, fixed_set<8>>(binomials, options);
		} else if (n <= 1024) {
			extn = count_linex_recursive<N, fixed_set<16>>(binomials, options);
		} else {
			extn = count_linex_recursive<N, dynamic_set>(binomials, options);
		}
		return extn;
	}
	
	
	
	
	
	
	template <typename N>
	struct Function
	{
		int domain_size;
		int range;
		int *domain;
		int n_values;
		N *values;
		
		int range_pow(int k)
		{
			int nn = 1;
			for (int i = 0; i < k; ++i) nn *= range;
			return nn;
		}
		
		Function(int domain_size, int range) : domain_size(domain_size), range(range)
		{
			domain = new int[domain_size];
			n_values = range_pow(domain_size);
			values = new N[n_values];
		}
		
		~Function()
		{
			delete [] domain;
			delete [] values;
		}
		
		int index(int *key)
		{
			int index = 0;
			int factor = 1;
			
			for (int i = 0; i < domain_size; ++i) {
				index += factor * key[i];
				factor *= range;
			}
			
			return index;
		}
		
		N &value(int *key)
		{
			return values[index(key)];
		}
		
		int map_index(int *key)
		{
			int index = 0;
			int factor = 1;
			
			for (int i = 0; i < domain_size; ++i) {
				index += factor * key[domain[i]];
				factor *= range;
			}
			
			return index;
		}
		
		N &map_value(int *map)
		{
			return values[map_index(map)];
		}
		
		bool domain_has(int k)
		{
			for (int i = 0; i < domain_size; ++i) {
				if (domain[i] == k) return true;
			}
			
			return false;
		}
		
		void print_domain()
		{
			if (domain_size == 0) {
				printf("Ø");
				return;
			}
			
			for (int i = 0; i < domain_size-1; ++i) printf("%i,", domain[i]);
			printf("%i", domain[domain_size-1]);
		}
		
		void print()
		{
			int nn = range_pow(domain_size);
			
			for (int i = 0; i < domain_size; ++i) printf("%i ", domain[i]);
			printf("\n");
			
			int *args = new int[domain_size+1];
			
			for (int i = 0; i < domain_size; ++i) args[i] = 0;
			
			for (int i = 0; i < nn; ++i) {
				for (int j = 0; j < domain_size; ++j) printf("%i ", args[j]);
				printf("  %i\n", value(args));
				
				int k = 0;
				while (++args[k] == range) args[k++] = 0;
			}
			
			delete [] args;
		}
	};
	
	
	template <typename N>
	Function<N> *make_arc_function(int u, int v, int range)
	{
		Function<N> *f = new Function<N>(2, range);
		f->domain[0] = u;
		f->domain[1] = v;
		
		int args[2];
		int &i = args[0];
		int &j = args[1];
		
		for (i = 0; i < range; ++i) {
			for (j = 0; j < range; ++j) {
				f->value(args) = i < j ? 1 : 0;
			}
		}
		
		return f;
	}
	
	template <typename N>
	Function<N> *eliminate(int var, Function<N> **functions, int n_functions, int range)
	{
		int n_variables = n;
		
		bool in_domain[n_variables];
		for (int i = 0; i < n_variables; ++i) in_domain[i] = false;
		
		for (int i = 0; i < n_functions; ++i) {
			for (int j = 0; j < functions[i]->domain_size; ++j) {
				in_domain[functions[i]->domain[j]] = true;
			}
		}
		
		int domain_size = 0;
		for (int i = 0; i < n_variables; ++i) {
			if (in_domain[i]) ++domain_size;
		}
		
		if (domain_size > induced_width) induced_width = domain_size;
		
		int *domain = new int[domain_size];
		
		int j = 0;
		for (int i = 0; i < n_variables; ++i) {
			if (in_domain[i]) domain[j++] = i;
		}
		
		
		Function<N> *f = new Function<N>(domain_size - 1, range);
		
		j = 0;
		for (int i = 0; i < n_variables; ++i) {
			if (in_domain[i] && i != var) f->domain[j++] = i;
		}
		
		int *key = new int[n_variables];
		for (int i = 0; i < domain_size; ++i) key[domain[i]] = 0;
		
		delete [] domain;
		
		for (int i = 0; i < f->n_values; ++i) {
			N sum = 0;
			
			for (key[var] = 0; key[var] < range; ++key[var]) {
				N product = 1;
				for (int j = 0; j < n_functions; ++j) product *= functions[j]->map_value(key);
				sum += product;
			}
			
			f->map_value(key) = sum;
			
			int k = 0;
			while (k < f->domain_size && ++key[f->domain[k]] == range) {
				key[f->domain[k++]] = 0;
			}
		}
		
		delete [] key;
		
		return f;
	}
	
	
	template <typename N>
	N elimination(Function<N> **functions, int n_functions, int *elim_order, int range)
	{
		int n_variables = n;
		
		Function<N> **remainder = new Function<N>*[n_functions];
		Function<N> **bucket = new Function<N>*[n_functions];
		
		for (int i = 0; i < n_functions; ++i) {
			remainder[i] = functions[i];
		}
		
		N empty = 1;
		
		for (int k = 0; k < n_variables; ++k) {
			eprintf(".");
			int var = elim_order[k];
			int bucket_size = 0;
			int remainder_size = 0;
			for (int i = 0; i < n_functions; ++i) {
				if (remainder[i]->domain_has(var)) {
					bucket[bucket_size++] = remainder[i];
				} else {
					remainder[remainder_size++] = remainder[i];
				}
			}
			
			if (bucket_size == 0) {
				empty *= range;
				continue;
			}
			
			Function<N> *f = eliminate<N>(var, bucket, bucket_size, range);
			
			for (int i = 0; i < bucket_size; ++i) {
				delete bucket[i];
			}
			
			remainder[remainder_size++] = f;
			n_functions = remainder_size;
		}
		
		for (int i = 0; i < n_functions; ++i) {
			assert(remainder[i]->domain_size == 0);
			empty *= remainder[i]->value(NULL);
		}
		
		for (int i = 0; i < n_functions; ++i) {
			delete remainder[i];
		}
		
		delete [] remainder;
		delete [] bucket;
		
		return empty;
	}
	
	template <typename N>
	N count_extensions(N **binomials, LECOptions &options)
	{
		std::vector<unsigned> order;
		
		int elim_order[n];
		
		if (options.elim_order_file != NULL) {
			FILE *f = fopen(options.elim_order_file, "r");
			assert(f);
			for (unsigned i = 0; i < n; ++i) assert(fscanf(f, "%i", &elim_order[i]) == 1);
			fclose(f);
		} else if (options.elimination_order == ELIM_ORDER_DEFAULT) {
			for (unsigned i = 0; i < n; ++i) elim_order[i] = i;
		} else if (options.elimination_order == ELIM_ORDER_REVERSE) {
			for (unsigned i = 0; i < n; ++i) elim_order[i] = n-i-1;
		} else if (options.elimination_order == ELIM_ORDER_LEAST_DEGREE) {
			find_least_degree_ordering(order);
			for (unsigned i = 0; i < n; ++i) elim_order[i] = order[i];
		} else if (options.elimination_order == ELIM_ORDER_TOPOLOGICAL) {
			find_tree_ordering(order);
			for (unsigned i = 0; i < n; ++i) elim_order[i] = order[i];
		} else {
			assert(0);
		}
		
		eprintf("Elimination order:");
		for (unsigned i = 0; i < n; ++i) {
			eprintf(" %i", elim_order[i]);
		}
		eprintf("\n");
		
		induced_width = 0;
		
		N ext = 0;
		N sign = n % 2 == 0 ? -1 : 1;
		for (unsigned k = 1; k <= n; ++k) {
			eprintf("Round %i ", k);
			Function<N> *functions[n*n];
			int n_functions = 0;
			for (unsigned u = 0; u < n; ++u) {
				for (unsigned v = 0; v < n; ++v) {
					if (has(u, v)) functions[n_functions++] = make_arc_function<N>(u, v, k);
				}
			}
			
			N elim = elimination<N>(functions, n_functions, elim_order, k);
			N binom = (n-k < k) ? binomials[n][n-k] : binomials[n][k];
			
			ext += sign * binom * elim;
			sign = -sign;
			eprintf("\n");
			if (k == 1) eprintf("Induced width: %i\n", induced_width);
		}
		
		return ext;
	}
	
};



template <typename N>
struct LinearExtensionCounterAuto
{
	unsigned n;
	N **binomials;
	LECOptions options;
	
	LinearExtensionCounterAuto(unsigned n) : n(n)
	{
		binomials = new N*[n+1];
		for (unsigned i = 0; i <= n; ++i) {
			binomials[i] = new N[i/2 + 1];
			binomials[i][0] = 1;
			for (unsigned j = 1; j <= i/2; ++j) {
				binomials[i][j] = binom(i-1, j-1) + binom(i-1, j);
			}
		}
	}
	
	~LinearExtensionCounterAuto()
	{
		for (unsigned i = 0; i <= n; ++i) {
			delete [] binomials[i];
		}
		delete [] binomials;
	}
	
	N binom(unsigned n, unsigned k) const
	{
		if (n-k < k) k = n-k;
		return binomials[n][k];
	}
	
	N count(const bool *matrix)
	{
		LinearExtensionCounter dg(n);
		for (unsigned j = 0; j < n; ++j) {
			for (unsigned i = 0; i < n; ++i) {
				if (matrix[j*n + i]) dg.flip(i, j);
			}
		}
		
		return dg.count_linex<N>(binomials, options);
	}
};





template <typename T>
struct LinearExtensionCounterAutoAccuracy
{
	unsigned n;
	LinearExtensionCounterAuto<long long int> *lec_llint;
	LinearExtensionCounterAuto<double> *lec_double;
	LinearExtensionCounterAuto<T> *lec_T;
	LECOptions options;
	
	LinearExtensionCounterAutoAccuracy(unsigned n) : n(n)
	{
		if (n <= 20) {
			lec_llint = new LinearExtensionCounterAuto<long long int>(n);
		} else if (n <= 170) {
			lec_double = new LinearExtensionCounterAuto<double>(n);
		} else {
			lec_T = new LinearExtensionCounterAuto<T>(n);
		}
	}
	
	~LinearExtensionCounterAutoAccuracy()
	{
		if (n <= 20) {
			delete lec_llint;
		} else if (n <= 170) {
			delete lec_double;
		} else {
			delete lec_T;
		}
	}
	
	T count(const bool *matrix)
	{
		T value;
		
		if (n <= 20) {
			lec_llint->options = options;
			value = (unsigned long long int)lec_llint->count(matrix);
		} else if (n <= 170) {
			lec_double->options = options;
			value = lec_double->count(matrix);
		} else {
			lec_T->options = options;
			value = lec_T->count(matrix);
		}
		
		return value;
	}
};

#endif
