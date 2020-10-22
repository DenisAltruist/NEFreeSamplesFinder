// g++ main.cc -O3 -o main -I/usr/include/python3.6 -lpython3.6m
// g++ main.cc -O3 -std=c++17 -o main -lpthread -I/usr/include/python3.6 -lpython3.6m

#include <bits/stdc++.h>
#include <condition_variable>
#include "thread_pool.h"

using namespace std;

const int kEdgeCostLimit = 25;

mt19937 mt(1);

int PlusInf() {
  return numeric_limits<int>::max();
}

int MinusInf() {
  return numeric_limits<int>::lowest();
}

template <typename T>
bool IsIntersection(const vector<T>& lhs, const vector<T>& rhs) {
  set<T> s;
  for (T x : lhs) {
    s.insert(x);
  }
  for (T x : rhs) {
    auto it = s.find(x);
    if (it != s.end()) {
      return true;
    }
  }
  return false;
}

template <typename Container>
void Shuffle(Container* c) {
  shuffle(c->begin(), c->end(), mt);
}

// inside [l; r]
int GetRandomInt(int l, int r) {
  assert(l <= r);
  size_t next_token = mt();
  int len = r - l + 1;
  return l + next_token % len;
}

void GenAllPossibleChoicesRec(const vector<int>& limits, vector<int>* cur_cortege, vector<vector<int>>* res) {
  size_t cur_idx = cur_cortege->size();
  if (cur_idx == limits.size()) {
    res->emplace_back(*cur_cortege);
    return;
  }
  for (size_t cur_elem = 0; cur_elem < limits[cur_idx]; cur_elem++) {
    cur_cortege->emplace_back(cur_elem);
    GenAllPossibleChoicesRec(limits, cur_cortege, res);
    cur_cortege->pop_back();
  }
}

vector<vector<int>> GenAllPossibleChoices(const vector<int>& limits) {
  vector<vector<int>> res;
  vector<int> cortege;
  GenAllPossibleChoicesRec(limits, &cortege, &res);
  return res;
}

void PrintCosts(const vector<int>& costs) {
  for (int cost : costs) {
    if (cost == PlusInf()) {
      cout << "+inf ";
    } else {
      cout << cost << " ";
    }
  }
  cout << endl;
}

struct Edge {
  int start;
  int finish;
  vector<int> cost;
  int idx;
};

class LPSolver {
 public:
  static void LaunchPython() {
  }

  static void ReleasePython() {
  }

  LPSolver() = default;

  LPSolver(int num_of_variables) : num_of_variables_(num_of_variables) {
  }

  pair<bool, vector<double>> GetSolution() {
    vector<double> res_list;
    return make_pair(false, vector<double>());
  }

  vector<int> GetIntSolution() {
    vector<int> res(num_of_variables_);
    auto sol_pair = GetSolution();

    vector<double> reg_sol = sol_pair.second;

    assert(sol_pair.first == true);
    for (int mask = 0; mask < (1 << num_of_variables_); mask++) {
      vector<double> trial(num_of_variables_);
      for (int bit_pos = num_of_variables_ - 1; bit_pos >= 0; --bit_pos) {
        int bit = (mask >> bit_pos) & 1;
        if (bit) {
          res[bit_pos] = int(reg_sol[bit_pos] + 1);
        } else {
          res[bit_pos] = int(reg_sol[bit_pos]);
        }
        trial[bit_pos] = res[bit_pos];
      }
      if (CheckSolution(trial)) {
        return res;
      }
    }
    assert(0);
  }

  bool CheckSolution(vector<double> sol) {
    assert(sol.size() == num_of_variables_);
    vector<double> sums(Size(), 0.0);
    bool is_ok = true;
    for (size_t ineq_idx = 0; ineq_idx < ineqs_.size(); ++ineq_idx) {
      for (size_t var_idx = 0; var_idx < num_of_variables_; ++var_idx) {
        sums[ineq_idx] += ineqs_[ineq_idx][var_idx] * sol[var_idx];
      }
    }

    for (size_t sum_idx = 0; sum_idx < sums.size(); ++sum_idx) {
      is_ok &= (sums[sum_idx] <= bounds_[sum_idx]);
    }
    return is_ok;
  }

  bool PushInequality(const vector<int>& ineq,
                      int bound) {  // last coeff is number after comparison sing. a_1x_1 + ... + a_nx_n <= c
    ineqs_.emplace_back(ineq);
    bounds_.emplace_back(bound);
    return true;
  }

  void Merge(const LPSolver& rhs) {
    for (size_t ineq_idx = 0; ineq_idx < rhs.ineqs_.size(); ++ineq_idx) {
      ineqs_.emplace_back(rhs.ineqs_[ineq_idx]);
      bounds_.emplace_back(rhs.bounds_[ineq_idx]);
    }
  }

  void PopInequality() {
    assert(!ineqs_.empty());
    ineqs_.pop_back();
    assert(!bounds_.empty());
    bounds_.pop_back();
  }

  size_t Size() {
    return bounds_.size();
  }

  void PrintIneqs() {
    for (size_t ineq_idx = 0; ineq_idx < ineqs_.size(); ++ineq_idx) {
      for (int x : ineqs_[ineq_idx]) {
        cout << x << " ";
      }
      cout << "<= " << bounds_[ineq_idx];
      cout << "\n";
    }
    cout << "\n";
  }

  bool IsFeasible() {
    if (!last_sol_.empty()) {
      bool res = CheckSolution(last_sol_);
      if (res) {
        return true;
      }
    }
    auto sol_pair = GetSolution();
    if (sol_pair.first) {
      last_sol_ = sol_pair.second;
    }
    return sol_pair.first;
  }

 private:
  int num_of_variables_;
  vector<int> bounds_;
  vector<vector<int>> ineqs_;
  vector<double> last_sol_;
};

void PrintVec(const vector<int>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    cout << v[i] << " ";
  }
  cout << endl;
}

struct SolverParameters {
  bool are_pay_costs_positive;  // all variables in linear systems are > 0
  // Len of acyclic part is [left_path_len_bound; right_path_len_bound]
  bool is_special_six_cycle_len_graph;
  int left_path_len_bound;
  int right_path_len_bound;
  int cycle_size;
  // for each vertex 'v' in acyclic part num_of_edges_to_cycle_bounds[v].first/second is [l;r] interval for number of
  // outs in cycle
  vector<pair<int, int>> num_of_edges_to_cycle_bounds;
  std::string offset_filename;       // name of file, where last checked cycle id would be stored
  bool should_shuffle_graphs;        // would shuffle order of graphs to check, if true
  bool need_to_remove_one_strategy;  // for test purposes

  mutex* log_mutex;
};

struct Path {
  vector<int> vertices;
};

struct HalfCycle {
  Path path[2];  // path[0] < path[1]
  vector<int> edge_idxs[2];

  uint64_t code;
  uint64_t rev_code;

  void Print() const {
    for (int x : path[0].vertices) {
      cout << x << " ";
    }
    cout << endl;
    for (int x : path[1].vertices) {
      cout << x << " ";
    }
    cout << endl;
    cout << endl;
  }

  void CalcCode() {
    code = 0;
    rev_code = 0;
    assert(path[0].vertices.size() >= 2);
    assert(path[1].vertices.size() >= 2);
    const int base = 47;  // prime hash base
    auto add_code = [&](const Path& path, uint64_t& val) -> void {
      for (int x : path.vertices) {
        val = val * base + x;
      }
    };
    add_code(path[0], code);
    add_code(path[1], code);
    add_code(path[1], rev_code);
    add_code(path[0], rev_code);
    if (path[0].vertices[1] > path[1].vertices[1]) {
      swap(code, rev_code);
    }
  }

  void Validate(const vector<int>& turns) const {
    assert(path[0].vertices.size() >= 2);
    assert(path[1].vertices.size() >= 2);
    assert(path[0].vertices[0] == path[1].vertices[0]);
    assert(path[0].vertices.back() == path[1].vertices.back());
    map<int, int> cntv;
    for (int path_idx = 0; path_idx < 2; ++path_idx) {
      set<int> path_set;  // to check that all vertices in path are distinct
      for (int v : path[path_idx].vertices) {
        assert(path_set.count(v) == 0);
        path_set.insert(v);
        cntv[v]++;
      }
    }
    int prev = -1;
    int turn_own = -1;
    for (int curv : path[0].vertices) {
      int cur_vertex_own = (cntv[curv] == 2);  // in intersection
      int prev_vertex_own = (cntv[prev] == 2);
      if (prev != -1 && prev_vertex_own == 1 && cur_vertex_own == 0) {
        assert(turns[prev] != -1);
        if (turn_own == -1) {
          turn_own = turns[prev];
        } else {
          assert(turn_own == turns[prev]);
        }
      }
      prev = curv;
    }
  }

  bool operator==(const HalfCycle& rhs) const {
    return code == rhs.code || code == rhs.rev_code;
  }
};

struct PathCollector {
  vector<vector<vector<Path>>> pathways_by_start_and_finish;
  int n;

  PathCollector(int n_val) : n(n_val) {
    pathways_by_start_and_finish = vector<vector<vector<Path>>>(n, vector<vector<Path>>(n));
  }

  pair<vector<HalfCycle>, vector<HalfCycle>> GetAllHalfCycles(const vector<int>& turns) const {
    vector<HalfCycle> res0, res1;
    for (int v = 0; v < n; ++v) {
      for (int u = 0; u < n; ++u) {
        if (v == u) {
          continue;
        }
        auto tmp_res = GetHalfCycles(v, u);
        for (const auto& P : tmp_res) {
          HalfCycle to_push;
          to_push.path[0] = pathways_by_start_and_finish[v][u][P.first];
          to_push.path[1] = pathways_by_start_and_finish[v][u][P.second];
          to_push.CalcCode();
          if (turns[v] == 0) {
            res0.emplace_back(to_push);
          } else if (turns[v] == 1) {
            res1.emplace_back(to_push);
          }
        }
      }
    }
    return make_pair(res0, res1);
  }

  vector<pair<int, int>> GetHalfCycles(int start, int finish) const {
    const vector<Path>& pathes = pathways_by_start_and_finish[start][finish];
    vector<pair<int, int>> res;
    for (size_t i = 0; i < pathes.size(); ++i) {
      for (size_t j = i + 1; j < pathes.size(); ++j) {
        vector<bool> is_used(n);

        auto check_path = [&](const Path& path) -> bool {
          bool res = false;
          for (int v : path.vertices) {
            if (v == start || v == finish) {
              continue;
            }
            res |= is_used[v];
            is_used[v] = true;
          }
          return res;
        };

        check_path(pathes[i]);
        bool are_intersected = check_path(pathes[j]);

        if (are_intersected) {
          continue;
        }
        res.emplace_back(make_pair(i, j));
      }
    }
    return res;
  }

  void CalcAllPathways(const vector<vector<int>>& M) {
    assert(n != 0);
    assert(M.size() == n);
    assert(M[0].size() == n);
    for (int start = 0; start < n; ++start) {
      for (int finish = 0; finish < n; ++finish) {
        if (start == finish) {
          continue;
        }
        for (int mask = 0; mask < (1 << n); ++mask) {
          int sbit = (mask >> start) & 1;
          int fbit = (mask >> finish) & 1;
          if (sbit || fbit) {
            continue;
          }
          vector<int> perm;
          for (int pos = n - 1; pos >= 0; --pos) {
            int bit = (mask >> pos) & 1;
            if (bit) {
              perm.emplace_back(pos);
            }
          }
          sort(perm.begin(), perm.end());
          do {
            vector<int> new_path_seq;
            int prev = start;
            bool is_correct_path = true;
            new_path_seq.emplace_back(start);
            for (int x : perm) {
              new_path_seq.emplace_back(x);
              if (!M[prev][x]) {
                is_correct_path = false;
                break;
              }
              prev = x;
            }
            is_correct_path &= M[prev][finish];
            if (is_correct_path) {
              new_path_seq.emplace_back(finish);
              Path new_path{.vertices = new_path_seq};
              pathways_by_start_and_finish[start][finish].emplace_back(new_path);
            }
          } while (next_permutation(perm.begin(), perm.end()));
        }
      }
    }
  }

  int GetPathwaysCnt() const {
    int res = 0;
    for (int v = 0; v < n; ++v) {
      for (int u = 0; u < n; ++u) {
        if (u == v) {
          continue;
        }
        res += pathways_by_start_and_finish[v][u].size();
      }
    }
    return res;
  }

  int GetPathwaysCnt(int v, int u) const {
    return pathways_by_start_and_finish[v][u].size();
  }
};

class NashDigraph {
 public:
  NashDigraph() = default;

  struct LinearFunc {
    vector<int> cycle_part;
    vector<int> acyclic_part;
    int64_t edges_mask;

    bool operator==(const LinearFunc& rhs) const {
      return edges_mask == rhs.edges_mask;
    }

    void UpdateEdgesMask() {
      edges_mask = 0;

      for (int edge_idx : cycle_part) {
        edges_mask += (1LL << edge_idx);
      }

      for (int edge_idx : acyclic_part) {
        edges_mask += (1LL << edge_idx);
      }
    }

    bool IsCycle() const {
      return !cycle_part.empty();
    }

    vector<int> GetVectorizedCycle(int vector_len, int var_sign) const {
      vector<int> res(vector_len);
      for (int edge_idx : cycle_part) {
        res[edge_idx] = var_sign;
      }
      return res;
    }

    vector<int> GetFullEdgesSet() const {
      vector<int> res = cycle_part;
      for (int edge_idx : acyclic_part) {
        res.emplace_back(edge_idx);
      }
      sort(res.begin(), res.end());
      return res;
    }

    void Print() const {
      cout << "cyclic part:";
      for (int x : cycle_part) {
        cout << " " << x;
      }
      cout << "\n";
      cout << "acyclic part:";
      for (int x : acyclic_part) {
        cout << " " << x;
      }
      cout << "\n\n";
    }
  };

  struct PathPosets {
    vector<vector<int>> graph;
    vector<pair<int, int>> edges;
    set<pair<int, int>> edges_set;

    vector<int> block_sizes;

    vector<int> dsu;
    vector<pair<int, int>> dsu_history;
    vector<int> set_size;
    vector<pair<int, int>> set_size_history;

    map<int64_t, int> path_idx_by_mask;
    map<pair<int64_t, int64_t>, vector<pair<int, int>>> path_pairs;
    int n, m;

    PathPosets() = default;

    PathPosets(const vector<LinearFunc>& paths, int edges_num) {
      dsu.resize(paths.size());
      set_size.resize(paths.size());
      graph.resize(paths.size());

      for (size_t i = 0; i < paths.size(); ++i) {
        dsu[i] = i;
        set_size[i] = 1;
      }

      int cur_path_idx = 0;
      for (const auto& path : paths) {
        if (path_idx_by_mask.count(path.edges_mask) > 0) {
          continue;
        }
        path_idx_by_mask[path.edges_mask] = cur_path_idx++;
      }

      m = edges_num;

      for (size_t i = 0; i < paths.size(); ++i) {
        for (size_t j = 0; j < paths.size(); ++j) {
          if (i == j) {
            continue;
          }

          int i_idx = path_idx_by_mask[paths[i].edges_mask];
          int j_idx = path_idx_by_mask[paths[j].edges_mask];

          path_pairs[BuildPathDiff(paths[i], paths[j])].emplace_back(i_idx, j_idx);
        }
      }

      n = cur_path_idx;
    }

    pair<int64_t, int64_t> BuildPathDiff(const LinearFunc& lhs, const LinearFunc& rhs) {
      int64_t i_mask = 0;
      int64_t j_mask = 0;
      int64_t xor_edges_mask = (lhs.edges_mask ^ rhs.edges_mask);
      for (int edge_idx = m - 1; edge_idx >= 0; --edge_idx) {
        int is_active = (xor_edges_mask >> edge_idx) & 1LL;
        if (!is_active) {
          continue;
        }
        int i_active = (lhs.edges_mask >> edge_idx) & 1;
        if (i_active) {
          i_mask += (1LL << edge_idx);
        } else {
          j_mask += (1LL << edge_idx);
        }
      }
      return make_pair(i_mask, j_mask);
    }

    size_t Size() {
      return edges.size();
    }

    void Rollback() {
      assert(!edges.empty());
      assert(!dsu_history.empty());
      assert(!set_size_history.empty());

      int v = edges.back().first;
      int u = edges.back().second;
      edges.pop_back();
      edges_set.erase(make_pair(v, u));

      assert(!graph[v].empty());
      graph[v].pop_back();

      auto pair = dsu_history.back();
      if (pair.first != -1) {
        dsu[pair.first] = pair.second;
      }
      dsu_history.pop_back();

      pair = set_size_history.back();
      if (pair.first != -1) {
        set_size[pair.first] = pair.second;
      }
      set_size_history.pop_back();
    }

    void dfs(int v, vector<bool>* is_vis) {
      (*is_vis)[v] = true;
      for (int u : graph[v]) {
        if (!(*is_vis)[u]) {
          dfs(u, is_vis);
        }
      }
    }

    bool HasPath(int v, int u) {
      vector<bool> is_vis(n);
      dfs(v, &is_vis);
      return is_vis[u];
    }

    bool AddBlock(const LinearFunc& v_path, const LinearFunc& u_path) {
      pair<int64_t, int64_t> path_diff = BuildPathDiff(v_path, u_path);

      int cnt = 0;
      bool is_ok = true;
      const auto& pairs = path_pairs[path_diff];
      for (const auto& edge : pairs) {
        if (edges_set.count(edge) > 0) {
          continue;
        }
        bool res = AddRawEdge(edge.first, edge.second);
        cnt += res;
        if (!res) {
          is_ok = false;
          break;
        }
      }
      if (is_ok) {
        block_sizes.emplace_back(cnt);
        return true;
      }
      while (cnt > 0) {
        --cnt;
        Rollback();
      }
      return false;
    }

    void PopBlock() {
      assert(!block_sizes.empty());
      int cnt = block_sizes.back();
      while (cnt > 0) {
        Rollback();
        cnt--;
      }
      block_sizes.pop_back();
    }

    bool AddEdge(const LinearFunc& v_path, const LinearFunc& u_path) {
      int v = path_idx_by_mask[v_path.edges_mask];
      int u = path_idx_by_mask[u_path.edges_mask];

      return AddRawEdge(u, v);
    }

    int GetRoot(int v) {
      if (v == dsu[v]) {
        return v;
      }
      return GetRoot(dsu[v]);
    }

    bool JoinSets(int v, int u) {
      v = GetRoot(v);
      u = GetRoot(u);
      if (v == u) {
        return false;
      }
      if (set_size[v] > set_size[u]) {
        swap(v, u);
      }
      dsu_history.emplace_back(v, dsu[v]);
      dsu[v] = u;
      set_size_history.emplace_back(v, set_size[v]);
      set_size[v] += set_size[u];
      return true;
    }

    bool AddRawEdge(int v, int u) {
      bool is_joined = false;
      // bool is_joined = JoinSets(v, u);
      if (is_joined || !HasPath(u, v)) {
        graph[v].emplace_back(u);
        edges.emplace_back(v, u);
        edges_set.insert(make_pair(v, u));

        /* dumb history */
        set_size_history.emplace_back(-1, -1);
        dsu_history.emplace_back(-1, -1);

        return true;
      }
      return false;
    }
  };

  NashDigraph(const string& path_to_file, bool is_complete) {
    is_complete_ = is_complete;
    ifstream in(path_to_file);
    assert(in.is_open());
    int num_of_vertices, num_of_edges, num_of_players;
    in >> num_of_vertices >> num_of_edges >> num_of_players >> start_vertex_;
    turns_ = vector<int>(num_of_vertices);
    edges_ = vector<vector<Edge>>(num_of_vertices);
    for (int vertex_idx = 0; vertex_idx < num_of_vertices; ++vertex_idx) {
      in >> turns_[vertex_idx];
      assert(turns_[vertex_idx] == -1 || (turns_[vertex_idx] >= 0 && turns_[vertex_idx] < num_of_vertices));
    }
    for (int edge_idx = 0; edge_idx < num_of_edges; ++edge_idx) {
      int v, u;
      in >> v >> u;
      assert(v >= 0 && v < num_of_vertices);
      assert(u >= 0 && u < num_of_vertices);
      vector<int> edge_cost(num_of_players);
      if (is_complete) {  // otherwise we will add (0, 0, ..., 0) costs
        for (int cost_idx = 0; cost_idx < num_of_players; ++cost_idx) {
          in >> edge_cost[cost_idx];
        }
      }
      AddEdge(v, u, edge_cost, edge_idx);
    }
    num_of_players_ = num_of_players;
    num_of_edges_ = num_of_edges;
    Preprocess(SolverParameters{});
  }

  // For this constructor edges should be added manually
  NashDigraph(const vector<int>& turns, int num_of_players, size_t start_vertex)
    : turns_(turns),
      edges_(vector<vector<Edge>>(turns.size())),
      start_vertex_(start_vertex),
      num_of_players_(num_of_players),
      num_of_edges_(0) {
  }

  void Preprocess(const SolverParameters& solver_params) {
    CalcAllPossiblePlayersStrategies(solver_params);
  }

  vector<vector<int>> GetAdjacentMatrix() const {
    int n = turns_.size();
    vector<vector<int>> res(n, vector<int>(n));
    for (int v = 0; v < n; ++v) {
      for (const auto& edge : edges_[v]) {
        res[v][edge.finish] = 1;
      }
    }
    return res;
  }

  vector<int> GetTurns() const {
    return turns_;
  }

  LPSolver ConfigureBaseLP(int player_num, const SolverParameters& solver_params) {
    LPSolver res(num_of_edges_);
    if (solver_params.are_pay_costs_positive) {
      for (size_t var_idx = 0; var_idx < num_of_edges_; ++var_idx) {
        vector<int> ineq(num_of_edges_);
        ineq[var_idx] = -1;
        res.PushInequality(ineq, -1);
      }
    }
    if (solver_params.is_special_six_cycle_len_graph) {
      // All pay-off costs are 1 on cycle edges
      std::vector<int> terminal_edge_idx_by_v(turns_.size());
      for (size_t v = 1; v <= 6; ++v) {
        for (auto& edge : edges_[v]) {
          if (edge.finish != 0) {
            vector<int> ineq(num_of_edges_);
            size_t var_idx = edge.idx;
            ineq[var_idx] = 1;
            res.PushInequality(ineq, 1);
          } else {
            terminal_edge_idx_by_v[v] = edge.idx;
          }
        }
      }
      /*
        Inequalities from papers on terminal edges, o1 - first player (0), o2 - second player (1)
        о1:  а_6 < а_5 < а_2 < а_1 < а_3 < а_4 < с ;
        о2:  а_3 < а_2 < а_6 < а_4 < а_5 < с;  а_6 < а_1 < с .
      */
      if (player_num != -1) {
        vector<vector<int>> chain_inequalities;
        if (player_num == 0) {
          chain_inequalities.emplace_back(vector<int>({6, 5, 2, 1, 3, 4}));
        } else {
          chain_inequalities.emplace_back(vector<int>({3, 2, 6, 4, 5}));
          chain_inequalities.emplace_back(vector<int>({6, 1}));
        }
        for (const auto& chain : chain_inequalities) {
          for (size_t i = 1; i < chain.size(); ++i) {
            vector<int> ineq(num_of_edges_);
            size_t var_idx_1 = terminal_edge_idx_by_v[chain[i - 1]];
            size_t var_idx_2 = terminal_edge_idx_by_v[chain[i]];
            ineq[var_idx_1] = 1;
            ineq[var_idx_2] = -1;
            res.PushInequality(ineq, -1);
          }
        }
      }
    }
    return res;
  }

  void CalcImprovementsTable(const SolverParameters& solver_params) {
    size_t n = all_possible_players_strategies_[0].size();
    size_t m = all_possible_players_strategies_[1].size();
    vector<vector<int>> can_improve_row(n, vector<int>(m, 0));
    vector<vector<int>> can_improve_col(n, vector<int>(m, 0));
    num_of_fails_by_cell_ = vector<vector<int>>(n, vector<int>(m, 0));
    vector<vector<LinearFunc>> linear_funcs_by_cell = GetLinearFuncsByCell();

    LPSolver base_lp_first_player = ConfigureBaseLP(0, solver_params);
    LPSolver base_lp_second_player = ConfigureBaseLP(1, solver_params);
    for (size_t cx = 0; cx < n; ++cx) {
      for (size_t cy = 0; cy < m; ++cy) {
        if (solver_params.are_pay_costs_positive && linear_funcs_by_cell[cx][cy].IsCycle()) {
          continue;
        }
        can_improve_row[cx][cy] = true;
        can_improve_col[cx][cy] = true;
      }
    }

    row_classes_ = vector<vector<int>>(n);
    for (int cx = 0; cx < n; ++cx) {
      map<vector<int>, int> occurs;
      for (int cy = 0; cy < m; ++cy) {
        occurs[linear_funcs_by_cell[cx][cy].GetFullEdgesSet()] = cy;
      }
      for (auto it = occurs.begin(); it != occurs.end(); ++it) {
        int cy = it->second;
        if (can_improve_row[cx][cy]) {
          row_classes_[cx].emplace_back(cy);
        }
      }
    }

    col_classes_ = vector<vector<int>>(m);
    for (int cy = 0; cy < m; ++cy) {
      map<vector<int>, int> occurs;

      for (int cx = 0; cx < n; ++cx) {
        occurs[linear_funcs_by_cell[cx][cy].GetFullEdgesSet()] = cx;
      }
      for (auto it = occurs.begin(); it != occurs.end(); ++it) {
        int cx = it->second;
        if (can_improve_col[cx][cy]) {
          col_classes_[cy].emplace_back(cx);
        }
      }
    }
  }

  void Print(bool with_costs) {
    cout << turns_.size() << " " << num_of_edges_ << " " << num_of_players_ << " " << start_vertex_ << endl;
    for (size_t turn_idx = 0; turn_idx < turns_.size(); ++turn_idx) {
      cout << turns_[turn_idx] << " ";
    }
    cout << endl;
    for (size_t v = 0; v < turns_.size(); ++v) {
      for (const Edge& e : edges_[v]) {
        cout << v << " " << e.finish << " " << e.idx;
        if (with_costs) {
          for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
            cout << " " << e.cost[player_idx];
          }
        }
        cout << endl;
      }
    }
  }

  void AddEmptyEdge(int v, int u, int edge_idx) {
    vector<int> mock_costs(num_of_players_);
    AddEdge(v, u, mock_costs, edge_idx);
  }

  void AddEdge(int v, int u, const vector<int>& costs, int edge_idx) {
    edges_[v].push_back(Edge{v, u, costs, edge_idx});
    num_of_edges_++;
  }

  void AddEdgeCosts(const vector<int>& src, vector<int>* dst) {
    for (size_t idx = 0; idx < src.size(); ++idx) {
      (*dst)[idx] += src[idx];
    }
  }

  void CalcPlayerStrategies(size_t player_idx) {
    size_t n = turns_.size();
    vector<int> player_num_of_edges_limits;
    for (size_t v = 0; v < n; ++v) {
      if (turns_[v] == static_cast<int>(player_idx)) {
        player_num_of_edges_limits.emplace_back(edges_[v].size());
      }
    }
    all_possible_players_strategies_[player_idx] = GenAllPossibleChoices(player_num_of_edges_limits);
  }

  void DfsRemoveStrategy(int v, map<int, int> edge_by_vertex, vector<int>* is_used) {
    (*is_used)[v] = true;
    if (turns_[v] == -1) {
      return;
    }

    auto it = edge_by_vertex.find(v);
    if (it != edge_by_vertex.end()) {
      int u = it->second;
      assert(u != v);
      DfsRemoveStrategy(u, edge_by_vertex, is_used);
      return;
    }

    for (const auto& edge : edges_[v]) {
      int u = edge.finish;
      if (!(*is_used)[u]) {
        assert(u != v);
        DfsRemoveStrategy(u, edge_by_vertex, is_used);
      }
    }
  }

  void CalcAllPossiblePlayersStrategies(const SolverParameters& solver_params) {
    all_possible_players_strategies_.resize(num_of_players_);
    for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
      CalcPlayerStrategies(player_idx);
    }

    if (!solver_params.need_to_remove_one_strategy) {
      return;
    }

    int player_idx = 0;

    PrintAllPlayerStrategies(player_idx);
    set<int> sts_to_ban;
    sts_to_ban.insert(0);

    const vector<int>& strategy = all_possible_players_strategies_[player_idx][0];

    map<int, int> edge_by_vertex;
    size_t st_idx = 0;
    for (size_t v = 0; v < turns_.size(); ++v) {
      if (turns_[v] == player_idx) {
        edge_by_vertex[v] = edges_[v][strategy[st_idx++]].finish;
      }
    }

    vector<int> is_used(turns_.size(), 0);

    DfsRemoveStrategy(start_vertex_, edge_by_vertex, &is_used);

    set<int> free_st_idxs;
    st_idx = 0;
    for (size_t v = 0; v < turns_.size(); ++v) {
      if (turns_[v] != player_idx) {
        continue;
      }

      if (!is_used[v]) {
        cerr << v << " " << st_idx << endl;
        free_st_idxs.insert(st_idx);
      }
      st_idx++;
    }

    for (size_t j = 1; j < all_possible_players_strategies_[player_idx].size(); ++j) {
      const vector<int>& st_sample = all_possible_players_strategies_[player_idx][j];
      bool is_same = true;
      for (size_t k = 0; k < st_sample.size(); ++k) {
        if (free_st_idxs.count(k) > 0) {
          continue;
        }
        if (st_sample[k] != strategy[k]) {
          is_same = false;
          break;
        }
      }
      if (is_same) {
        sts_to_ban.insert(j);
      }
    }
    vector<vector<int>> sifted_strategies;
    for (size_t j = 0; j < all_possible_players_strategies_[player_idx].size(); ++j) {
      if (sts_to_ban.count(j) > 0) {
        continue;
      }
      sifted_strategies.emplace_back(all_possible_players_strategies_[player_idx][j]);
    }
    all_possible_players_strategies_[player_idx] = sifted_strategies;
    PrintAllPlayerStrategies(player_idx);
  }

  void ApplyPlayerStrategyToGlobalOne(const vector<int>& player_strategy,
                                      size_t player_idx,
                                      vector<size_t>* all_players_strategy) {
    size_t n = turns_.size();
    size_t cur_player_edge_idx = 0;
    for (size_t v = 0; v < n; ++v) {
      if (turns_[v] == static_cast<int>(player_idx)) {
        (*all_players_strategy)[v] = player_strategy[cur_player_edge_idx++];
      }
    }
  }

  void SetRandomEdgeCosts(int l, int r) {
    size_t n = turns_.size();
    for (size_t v = 0; v < n; ++v) {
      for (auto& edge : edges_[v]) {
        for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
          edge.cost[player_idx] = GetRandomInt(l, r);
        }
      }
    }
  }

  void CheckCorrectness() {
    vector<pair<int, int>> sol = GetSolution();
    cout << "Solutions in format: (edge index, cost_1, cost_2)\n";
    vector<vector<int>> edge_cost_by_edge_idx(num_of_edges_);
    for (int edge_idx = 0; edge_idx < num_of_edges_; ++edge_idx) {
      edge_cost_by_edge_idx[edge_idx] = vector<int>{sol[edge_idx].first, sol[edge_idx].second};
    }

    for (size_t v = 0; v < turns_.size(); ++v) {
      for (auto& edge : edges_[v]) {
        edge.cost = edge_cost_by_edge_idx[edge.idx];
      }
    }

    is_complete_ = true;

    int n = all_possible_players_strategies_[0].size();
    int m = all_possible_players_strategies_[1].size();
    int num_of_ne = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        vector<size_t> all_player_strategy(turns_.size());
        ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][i], 0, &all_player_strategy);
        ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][j], 1, &all_player_strategy);
        bool is_pos_in_ne = IsStrategyNE(all_player_strategy);
        if (best_cells_cover_matrix_[i][j]) {
          assert(!is_pos_in_ne);
        }
        num_of_ne += is_pos_in_ne;
      }
    }
    double util = (n * m - num_of_ne) / (n * m + 0.0);
    cerr << fixed << setprecision(6) << util << " " << num_of_ne << endl;
    is_complete_ = false;
  }

  int CountNumOfNE() {
    auto lin_funcs = GetLinearFuncsByCell();
    size_t n = turns_.size();
    vector<int> num_of_strategies_limits(num_of_players_);
    for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
      num_of_strategies_limits[player_idx] = all_possible_players_strategies_[player_idx].size();
    }
    vector<vector<int>> all_possible_strategies_corteges = GenAllPossibleChoices(num_of_strategies_limits);
    int total_num_of_corteges = all_possible_strategies_corteges.size();
    int num_of_corteges_in_ne = 0;
    // cout << "Total num of tuples of strategies: " << total_num_of_corteges << endl;
    for (const vector<int>& strategy_cortege : all_possible_strategies_corteges) {
      vector<size_t> all_players_strategy(n);  // 0 will remain for terminals
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        size_t strategy_for_cur_player_to_use_idx = strategy_cortege[player_idx];
        const vector<int>& cur_player_strategy =
          all_possible_players_strategies_[player_idx][strategy_for_cur_player_to_use_idx];
        ApplyPlayerStrategyToGlobalOne(cur_player_strategy, player_idx, &all_players_strategy);
      }
      bool is_strategy_ne = IsStrategyNE(all_players_strategy);
      if (is_strategy_ne) {
        for (int v = 0; v < n; ++v) {
          if (turns_[v] == -1) {
            continue;
          }
          int edge_idx = all_players_strategy[v];
          cout << v << " " << edges_[v][edge_idx].finish << endl;
        }
      }
      num_of_corteges_in_ne += is_strategy_ne;
    }
    // cout << "Num of corteges in NE: " << num_of_corteges_in_ne << endl;
    assert(total_num_of_corteges != 0);
    return num_of_corteges_in_ne;
  }

  // strategy[i] is index of edge for vertex i, so, strategy[i] in [0; edges_[i].size());
  vector<int> CalcPlayersTotalSums(const vector<size_t>& strategy) {
    return CalcPlayersTotalSums(strategy, start_vertex_, false);
  }

  bool IsStrategyNE(const vector<size_t>& strategy) {
    vector<int> total_costs = CalcPlayersTotalSums(strategy);
    size_t n = strategy.size();
    for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
      vector<size_t> all_players_strategy = strategy;
      for (const vector<int>& changed_strategy : all_possible_players_strategies_[player_idx]) {
        ApplyPlayerStrategyToGlobalOne(changed_strategy, player_idx, &all_players_strategy);
        vector<int> new_player_total_sums = CalcPlayersTotalSums(all_players_strategy);
        // player 'player_idx' can improve their strategy, it's not NE
        if (new_player_total_sums[player_idx] < total_costs[player_idx]) {
          return false;
        }
      }
    }

    return true;
  }

  LinearFunc GetLinFuncForPlayersByGlobalStrategy(const vector<size_t>& all_players_strategy) {
    size_t curv = start_vertex_;
    vector<int> is_vertex_used(turns_.size());
    vector<int> used_edges;
    vector<pair<int, int>> edges_path;
    while (!is_vertex_used[curv] && turns_[curv] != -1) {
      is_vertex_used[curv] = 1;
      assert(curv < all_players_strategy.size());
      size_t index_of_edge_to_use = all_players_strategy[curv];
      assert(index_of_edge_to_use < edges_[curv].size());
      size_t nextv = edges_[curv][index_of_edge_to_use].finish;
      used_edges.emplace_back(edges_[curv][index_of_edge_to_use].idx);
      edges_path.emplace_back(make_pair(curv, index_of_edge_to_use));
      curv = nextv;
    }
    LinearFunc res;
    if (is_vertex_used[curv]) {  // got cycle
      bool is_already_in_cycle = (start_vertex_ == curv);
      for (const auto& edge_coords : edges_path) {
        const auto& edge = edges_[edge_coords.first][edge_coords.second];
        if (!is_already_in_cycle) {
          res.acyclic_part.emplace_back(edge.idx);
        } else {
          res.cycle_part.emplace_back(edge.idx);
        }
        if (edge.finish == curv) {
          is_already_in_cycle = true;
        }
      }
      return res;
    }
    res.acyclic_part = used_edges;
    return res;
  }

  // returns true if improved by addition
  bool AddInequality(const LinearFunc& old_func,
                     const LinearFunc& new_func,
                     const SolverParameters& solver_params,
                     PathPosets* path_posets) {
    if (old_func.IsCycle() || new_func.IsCycle()) {
      if (solver_params.are_pay_costs_positive) {
        if (old_func.IsCycle()) {
          if (!new_func.IsCycle()) {
            return true;
          }
          return false;
        }
        return false;
      }
      return true;
    }
    return path_posets->AddBlock(old_func, new_func);
  }

  void UpdateMH(int player, int64_t edges_mask, int delta) {
    MH[player][edges_mask] += delta;
    for (const int cord : cords_by_edges_mask[player][edges_mask]) {
      MH_cov[player][cord] += delta;
    }
  }

  bool CanAddMH(int player, int64_t edges_mask) {
    if (MH[player][edges_mask] > 0) {
      return true;  // already added
    }
    int opp = 1 - player;
    if (MH[opp][edges_mask] > 0) {
      return false;  // We would get intersection on MH sets
    }

    bool can_add = true;
    for (const int cord : cords_by_edges_mask[player][edges_mask]) {
      MH_cov[player][cord]++;
      if (MH_cov[player][cord] == MH_cov_max[player]) {
        can_add = false;
      }
    }
    for (const int cord : cords_by_edges_mask[player][edges_mask]) {
      MH_cov[player][cord]--;
    }

    return can_add;
  }

  bool GoFirstPlayer(const vector<vector<LinearFunc>>& linear_funcs_by_cell,
                     const SolverParameters& solver_params,
                     int cx,
                     int cy,
                     vector<vector<int>>* is_cell_used,
                     LPSolver* lp_x,
                     LPSolver* lp_y) {
    int n = is_cell_used->size();
    // finding cell to improve for the first player
    for (int tx : col_classes_[cy]) {
      int64_t edges_mask = linear_funcs_by_cell[tx][cy].edges_mask;
      if (!CanAddMH(0, edges_mask)) {
        continue;
      }
      size_t old_posets_size = path_posets_x_.Size();
      if (linear_funcs_by_cell[tx][cy] == linear_funcs_by_cell[cx][cy]) {
        continue;
      }
      const LinearFunc& best_linear_func = linear_funcs_by_cell[tx][cy];
      bool can_add_ineqs = true;
      for (int func_idx : col_classes_[cy]) {
        if (func_idx == tx) {
          continue;
        }
        can_add_ineqs &= AddInequality(
          linear_funcs_by_cell[func_idx][cy], linear_funcs_by_cell[tx][cy], solver_params, &path_posets_x_);
      }
      if (can_add_ineqs) {
        vector<int> colored_cells;
        for (int wx = 0; wx < n; ++wx) {
          if (linear_funcs_by_cell[wx][cy] == linear_funcs_by_cell[tx][cy]) {
            continue;
          }
          if (!(*is_cell_used)[wx][cy]) {
            colored_cells.emplace_back(wx);
            (*is_cell_used)[wx][cy] = 1;
          }
        }
        UpdateMH(0, edges_mask, 1);
        bool branch_result =
          SolveTwoPlayersCostsRec(linear_funcs_by_cell, solver_params, tx, cy, 1, is_cell_used, lp_x, lp_y);
        if (branch_result) {
          return true;
        }
        UpdateMH(0, edges_mask, -1);
        for (int colored_cell : colored_cells) {
          (*is_cell_used)[colored_cell][cy] = 0;
        }
      }

      while (path_posets_x_.Size() != old_posets_size) {
        path_posets_x_.PopBlock();
      }
    }
    return false;
  }

  bool GoSecondPlayer(const vector<vector<LinearFunc>>& linear_funcs_by_cell,
                      const SolverParameters& solver_params,
                      int cx,
                      int cy,
                      vector<vector<int>>* is_cell_used,
                      LPSolver* lp_x,
                      LPSolver* lp_y) {
    int m = (*is_cell_used)[0].size();
    // finding cell to improve for the second player
    for (int ty : row_classes_[cx]) {
      int64_t edges_mask = linear_funcs_by_cell[cx][ty].edges_mask;
      if (!CanAddMH(1, edges_mask)) {
        continue;
      }
      size_t old_posets_size = path_posets_y_.Size();
      if (linear_funcs_by_cell[cx][ty] == linear_funcs_by_cell[cx][cy]) {
        continue;
      }
      const LinearFunc& best_linear_func = linear_funcs_by_cell[cx][ty];
      bool can_add_ineqs = true;
      for (int func_idx : row_classes_[cx]) {
        if (func_idx == ty) {
          continue;
        }
        can_add_ineqs &= AddInequality(
          linear_funcs_by_cell[cx][func_idx], linear_funcs_by_cell[cx][ty], solver_params, &path_posets_y_);
      }
      if (can_add_ineqs) {
        vector<int> colored_cells;
        for (int wy = 0; wy < m; ++wy) {
          if (linear_funcs_by_cell[cx][wy] == linear_funcs_by_cell[cx][ty]) {
            continue;
          }
          if (!(*is_cell_used)[cx][wy]) {
            colored_cells.emplace_back(wy);
            (*is_cell_used)[cx][wy] = 1;
          }
        }
        UpdateMH(1, edges_mask, 1);
        bool branch_result =
          SolveTwoPlayersCostsRec(linear_funcs_by_cell, solver_params, cx, ty, 0, is_cell_used, lp_x, lp_y);
        if (branch_result) {
          return true;
        }
        UpdateMH(1, edges_mask, -1);
        for (int colored_cell : colored_cells) {
          (*is_cell_used)[cx][colored_cell] = 0;
        }
      }
      while (path_posets_y_.Size() != old_posets_size) {
        path_posets_y_.PopBlock();
      }
    }
    return false;
  }

  bool SolveTwoPlayersCostsRec(const vector<vector<LinearFunc>>& linear_funcs_by_cell,
                               const SolverParameters& solver_params,
                               int cx,
                               int cy,
                               int direction,  // -1 for first cell, 0 - GoFirstPlayer, 1 - GoSecondPlayer
                               vector<vector<int>>* is_cell_used,
                               LPSolver* lp_x,
                               LPSolver* lp_y) {
    int n = is_cell_used->size();
    int m = (*is_cell_used)[0].size();
    assert(cx < n);
    assert(cy < m);
    if ((*is_cell_used)[cx][cy]) {
      int wx, wy, max_num_of_fails = -1;
      int num_of_used_cells = 0;
      for (int tx = 0; tx < n; ++tx) {
        for (int ty = 0; ty < m; ++ty) {
          if (!(*is_cell_used)[tx][ty]) {
            if (num_of_fails_by_cell_[tx][ty] > max_num_of_fails) {
              max_num_of_fails = num_of_fails_by_cell_[tx][ty];
              wx = tx;
              wy = ty;
            }
          } else {
            num_of_used_cells++;
          }
        }
      }
      if (max_num_of_fails == -1) {
        return true;
      }
      double sat_percentage_ = double(num_of_used_cells) / (n * m);
      // cerr << "Sat percentage: " << sat_percentage_ << std::endl;
      if (sat_percentage_ > ineq_sat_percentage_) {
        // cerr << "Updated sat percentage: " << sat_percentage_ << std::endl;
        ineq_sat_percentage_ = sat_percentage_;
        best_solver_first_player_ = *lp_x;
        best_solver_second_player_ = *lp_y;
        best_cells_cover_matrix_ = *is_cell_used;
      }
      return SolveTwoPlayersCostsRec(linear_funcs_by_cell, solver_params, wx, wy, -1, is_cell_used, lp_x, lp_y);
    }
    (*is_cell_used)[cx][cy] = 1;
    // randomizing branch's order

    if (direction == -1) {
      bool res = GoFirstPlayer(linear_funcs_by_cell, solver_params, cx, cy, is_cell_used, lp_x, lp_y);
      if (res) {
        return true;
      }
      res = GoSecondPlayer(linear_funcs_by_cell, solver_params, cx, cy, is_cell_used, lp_x, lp_y);
      if (res) {
        return true;
      }
      num_of_fails_by_cell_[cx][cy]++;
      (*is_cell_used)[cx][cy] = 0;
      return false;
    }

    if (direction == 0) {
      bool res = GoFirstPlayer(linear_funcs_by_cell, solver_params, cx, cy, is_cell_used, lp_x, lp_y);
      if (res) {
        return true;
      }
      num_of_fails_by_cell_[cx][cy]++;
      (*is_cell_used)[cx][cy] = 0;
      return false;
    }

    bool res = GoSecondPlayer(linear_funcs_by_cell, solver_params, cx, cy, is_cell_used, lp_x, lp_y);
    if (res) {
      return true;
    }
    num_of_fails_by_cell_[cx][cy]++;
    (*is_cell_used)[cx][cy] = 0;
    return false;
  }

  double GetIneqSatPercentage() {
    return ineq_sat_percentage_;
  }

  void PrintAllPlayerStrategies(int player_num) {
    vector<int> own_player_vectices;
    for (size_t v = 0; v < turns_.size(); ++v) {
      if (turns_[v] == player_num) {
        own_player_vectices.emplace_back(v);
      }
    }

    for (const vector<int>& strategy : all_possible_players_strategies_[player_num]) {
      ostringstream ostr;
      assert(strategy.size() == own_player_vectices.size());
      for (size_t vertex_idx = 0; vertex_idx < strategy.size(); ++vertex_idx) {
        int v = own_player_vectices[vertex_idx];
        const auto& edge = edges_[v][strategy[vertex_idx]];
        if (vertex_idx != 0) {
          ostr << " | ";
        }
        ostr << v << "->" << edge.finish;
      }
      cout << ostr.str() << endl;
    }
  }

  vector<vector<LinearFunc>> GetLinearFuncsByCell() {
    int n = all_possible_players_strategies_[0].size();
    int m = all_possible_players_strategies_[1].size();
    vector<vector<LinearFunc>> res(n, vector<LinearFunc>(m));
    for (int cx = 0; cx < n; ++cx) {
      for (int cy = 0; cy < m; ++cy) {
        vector<size_t> all_players_strategy(turns_.size());
        ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][cx], 0, &all_players_strategy);
        ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][cy], 1, &all_players_strategy);
        res[cx][cy] = GetLinFuncForPlayersByGlobalStrategy(all_players_strategy);
        res[cx][cy].UpdateEdgesMask();
      }
    }
    return res;
  }

  bool SolveTwoPlayersCosts(const SolverParameters& solver_params) {
    assert(num_of_players_ == 2);
    int n = all_possible_players_strategies_[0].size();
    int m = all_possible_players_strategies_[1].size();
    ineq_sat_percentage_ = -1.0;
    {
      std::unique_lock<std::mutex> lock(*solver_params.log_mutex);
      cout << "Num of strategies for players: " << n << " " << m << endl;
    }
    if (n == 0 || m == 0) {
      return false;
    }
    vector<vector<LinearFunc>> linear_funcs_by_cell = GetLinearFuncsByCell();

    vector<LinearFunc> flatten_lin_funcs;
    for (int cx = 0; cx < n; ++cx) {
      for (int cy = 0; cy < m; ++cy) {
        if (linear_funcs_by_cell[cx][cy].IsCycle()) {
          continue;
        }
        flatten_lin_funcs.emplace_back(linear_funcs_by_cell[cx][cy]);
      }
    }

    path_posets_x_ = PathPosets(flatten_lin_funcs, num_of_edges_);
    path_posets_y_ = path_posets_x_;

    vector<vector<int>> is_pair_of_strategies_used(n, vector<int>(m));
    int sx = -1, sy = -1;

    MH_cov[1] = vector<int>(m, 0);
    MH_cov[0] = vector<int>(n, 0);
    MH_cov_max[1] = n;
    MH_cov_max[0] = m;

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (!linear_funcs_by_cell[i][j].IsCycle()) {
          sx = i;
          sy = j;
          int64_t edges_mask = linear_funcs_by_cell[i][j].edges_mask;
          cords_by_edges_mask[0][edges_mask].emplace_back(i);
          cords_by_edges_mask[1][edges_mask].emplace_back(j);
        }
      }
    }

    LPSolver lp_x = ConfigureBaseLP(0, solver_params);
    LPSolver lp_y = ConfigureBaseLP(1, solver_params);

    return SolveTwoPlayersCostsRec(
      linear_funcs_by_cell, solver_params, sx, sy, -1, &is_pair_of_strategies_used, &lp_x, &lp_y);
  }

  void Dfs(int cur_vertex, vector<int>* is_vertex_visited) {
    (*is_vertex_visited)[cur_vertex] = 1;
    for (const Edge& edge : edges_[cur_vertex]) {
      int next_vertex = edge.finish;
      if (!(*is_vertex_visited)[next_vertex]) {
        Dfs(next_vertex, is_vertex_visited);
      }
    }
  }

  vector<pair<int, int>> GetSolution() {
    vector<pair<int, int>> res;
    vector<int> first_player_sol = best_solver_first_player_.GetIntSolution();
    vector<int> second_player_sol = best_solver_second_player_.GetIntSolution();
    for (size_t var_idx = 0; var_idx < first_player_sol.size(); ++var_idx) {
      res.emplace_back(first_player_sol[var_idx], second_player_sol[var_idx]);
    }
    return res;
  }

  bool AreAllVerticesAccessibleFromStart() {
    int n = turns_.size();
    vector<int> is_vertex_visited(n);
    Dfs(start_vertex_, &is_vertex_visited);
    for (int v = 0; v < n; ++v) {
      if (!is_vertex_visited[v]) {
        return false;
      }
    }
    return true;
  }

  bool HasOnlyOneTerminal() {
    int num_of_terminals = 0;
    int n = turns_.size();
    for (int v = 0; v < n; ++v) {
      if (edges_[v].size() == 0) {
        num_of_terminals += 1;
      }
    }
    return num_of_terminals == 1;
  }

  bool AreAllVerticesHasAtLeastTwoMoves() {
    int n = turns_.size();
    for (int v = 0; v < n; ++v) {
      if (turns_[v] == -1) {
        continue;
      }
      if (edges_[v].size() < 2) {
        return false;
      }
    }
    return true;
  }

  vector<int> GetPassport() const {
    int n = turns_.size();
    vector<int> res(2 * n);
    for (int v = 0; v < n; ++v) {
      for (const auto& edge : edges_[v]) {
        res[v]++;
        res[n + edge.finish]++;
      }
    }
    sort(res.begin(), res.begin() + n);
    sort(res.begin() + n, res.begin() + 2 * n);
    return res;
  }

  void CheckHypoNew() {
    int n = all_possible_players_strategies_[0].size();
    int m = all_possible_players_strategies_[1].size();
    vector<vector<LinearFunc>> funcs = GetLinearFuncsByCell();

    vector<vector<int>> costs_by_edge_idx(num_of_edges_);

    int vnum = turns_.size();
    for (int v = 0; v < vnum; ++v) {
      for (const Edge& e : edges_[v]) {
        costs_by_edge_idx[e.idx] = e.cost;
      }
    }

    set<vector<int>> MH, HM;

    map<int, vector<int>> outc;
    for (int i = 0; i < n; ++i) {
      int cur_min = 1e9;
      vector<vector<int>> best_edges_set;
      for (int j = 0; j < m; ++j) {
        vector<int> edges_set = funcs[i][j].GetFullEdgesSet();

        if (funcs[i][j].IsCycle()) {
          continue;
        }
        int cur_cost = 0;
        for (int edge_idx : edges_set) {
          cur_cost += costs_by_edge_idx[edge_idx][1];
        }
        /*
        auto kek = outc.find(cur_cost);
        if (kek == outc.end()) {
          outc[cur_cost] = edges_set;
        } else if (kek->second != edges_set) {
          cout << "TWO DIFF PARTIES WITH SAME COST" << endl;
          cout << i << " " << j << endl;
        }
        */

        if (cur_min > cur_cost) {
          cur_min = cur_cost;
          best_edges_set.clear();
          best_edges_set.emplace_back(edges_set);
        } else if (cur_min == cur_cost) {
          bool is_found = false;
          for (const auto& cur_edges_set : best_edges_set) {
            if (cur_edges_set == edges_set) {
              is_found = true;
              break;
            }
          }
          if (!is_found) {
            best_edges_set.emplace_back(edges_set);
          }
        }
      }
      for (const auto& edges_set : best_edges_set) {
        MH.insert(edges_set);
      }
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        vector<int> edges_set = funcs[i][j].GetFullEdgesSet();
        if (funcs[i][j].IsCycle()) {
          continue;
        }
        auto it = MH.find(edges_set);
        if (it == MH.end()) {
          HM.insert(edges_set);
        }
      }
    }
    bool is_mh_strat_found = false;
    for (int j = 0; j < m; ++j) {
      bool is_ok = true;
      for (int i = 0; i < n; ++i) {
        vector<int> edges_set = funcs[i][j].GetFullEdgesSet();
        auto it = MH.find(edges_set);
        if (it == MH.end()) {
          is_ok = false;
          break;
        }
      }
      if (is_ok) {
        is_mh_strat_found = true;
        break;
      }
    }
    if (!is_mh_strat_found) {
      cerr << "BAD MH!" << endl;
      exit(0);
    }
    LPSolver lp(num_of_edges_);
    for (auto it = MH.begin(); it != MH.end(); ++it) {
      const vector<int>& lhs = *it;
      for (auto jt = HM.begin(); jt != HM.end(); ++jt) {
        const vector<int>& rhs = *jt;
        vector<int> ineq(num_of_edges_);
        for (int var_idx : lhs) {
          ineq[var_idx] += 1;
        }
        for (int var_idx : rhs) {
          ineq[var_idx] -= 1;
        }
        lp.PushInequality(ineq, -1);
      }
    }
    if (lp.Size() == 0) {
      cout << "EMPTY! Continue ..." << endl;
      return;
    }
    bool is_feasible = lp.IsFeasible();

    if (!is_feasible) {
      cout << "BAD!" << endl;
      lp.PrintIneqs();
      Print(true);
      exit(0);
    }
  }

  vector<pair<int, int>> CheckHypo(int player_idx) {
    int n = all_possible_players_strategies_[0].size();
    int m = all_possible_players_strategies_[1].size();

    vector<vector<LinearFunc>> funcs = GetLinearFuncsByCell();
    struct Outcome {
      vector<int> cost;
      bool is_cycle;
      int num;

      Outcome() {
        cost = vector<int>(2);
      }
    };

    vector<vector<Outcome>> outcomes(n, vector<Outcome>(m));

    vector<vector<int>> costs_by_edge_idx(num_of_edges_);

    int vnum = turns_.size();
    for (int v = 0; v < vnum; ++v) {
      for (const Edge& e : edges_[v]) {
        costs_by_edge_idx[e.idx] = e.cost;
      }
    }
    vector<pair<int, int>> cycles;

    vector<pair<int, pair<int, int>>> lab_outc;
    if (player_idx == 0) {
      vector<vector<Outcome>> transp_mat(m, vector<Outcome>(n));
      vector<vector<LinearFunc>> transp_funcs(m, vector<LinearFunc>(n));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
          transp_mat[j][i] = outcomes[i][j];
          transp_funcs[j][i] = funcs[i][j];
        }
      }
      outcomes = transp_mat;
      funcs = transp_funcs;
      swap(n, m);
      for (int edge_idx = 0; edge_idx < num_of_edges_; ++edge_idx) {
        reverse(costs_by_edge_idx[edge_idx].begin(), costs_by_edge_idx[edge_idx].end());
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (funcs[i][j].IsCycle()) {
          cycles.emplace_back(i, j);
          continue;
        }
        vector<int> edges_set = funcs[i][j].GetFullEdgesSet();
        outcomes[i][j].is_cycle = false;

        for (int edge_idx : edges_set) {
          for (int k = 0; k < 2; ++k) {
            outcomes[i][j].cost[k] += costs_by_edge_idx[edge_idx][k];
          }
        }
        lab_outc.emplace_back(make_pair(outcomes[i][j].cost[1], make_pair(i, j)));
      }
    }

    int num_of_outcomes = 0;
    sort(lab_outc.begin(), lab_outc.end());
    map<vector<int>, int> set_num;
    for (const auto& outc : lab_outc) {
      size_t i = outc.second.first;
      size_t j = outc.second.second;
      vector<int> edges_set = funcs[i][j].GetFullEdgesSet();
      auto it = set_num.find(edges_set);
      if (it != set_num.end()) {
        outcomes[i][j].num = it->second;
      } else {
        outcomes[i][j].num = num_of_outcomes;
        set_num[edges_set] = num_of_outcomes++;
      }
    }

    num_of_outcomes++;

    for (auto& cycle : cycles) {
      outcomes[cycle.first][cycle.second].is_cycle = true;
      outcomes[cycle.first][cycle.second].num = num_of_outcomes;
    }

    auto calc_mn = [&](const vector<Outcome>& str, int* idx) -> int {
      int mn = numeric_limits<int>::max();
      for (size_t i = 0; i < str.size(); ++i) {
        const auto& outcome = str[i];
        if (!outcome.is_cycle) {
          if (outcome.cost[1] < mn) {
            *idx = i;
            mn = outcome.cost[1];
          }
        }
      }
      return mn;
    };

    std::sort(
      outcomes.begin(), outcomes.end(), [&](const vector<Outcome>& lhs_str, const vector<Outcome>& rhs_str) -> bool {
        int x;
        int lhs_mn = calc_mn(lhs_str, &x);
        int rhs_mn = calc_mn(rhs_str, &x);
        return lhs_mn < rhs_mn;
      });

    auto get_lex_code = [&](int idx) -> vector<int> {
      vector<int> code(num_of_outcomes);
      for (int i = 0; i < n; ++i) {
        code[num_of_outcomes - outcomes[i][idx].num] = 1;
      }
      return code;
    };

    int best = 0;
    for (int j = 1; j < m; ++j) {
      if (get_lex_code(best) > get_lex_code(j)) {
        best = j;
      }
    }

    auto print = [&](const vector<int>& code) {
      for (int x : code) {
        cerr << x;
      }
      cerr << endl;
    };

    // print(get_lex_code(best));

    vector<pair<int, int>> cells_to_check;

    if (player_idx == 1) {
      for (int i = 0; i < n; ++i) {
        cells_to_check.emplace_back(i, best);
      }
    } else {
      swap(n, m);
      for (int j = 0; j < m; ++j) {
        cells_to_check.emplace_back(best, j);
      }
    }
    vector<pair<int, int>> res;
    for (const auto& cell : cells_to_check) {
      size_t i = cell.first;
      size_t j = cell.second;
      vector<size_t> all_players_strategy(turns_.size());  // 0 will remain for terminals
      ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][i], 0, &all_players_strategy);
      ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][j], 1, &all_players_strategy);
      if (IsStrategyNE(all_players_strategy)) {
        res.emplace_back(i, j);
      }
    }
    return res;
  }

 private:
  struct HalfCycleVar {
    int player_num;  // == 0 => 'half_cycles_cx_', == 1 => 'half_cycles_cy_'
    int index;       // index in 'half_cycles_cx_' or 'half_cycles_cy_'
    int sign;        // 0, 1
  };

  PathPosets path_posets_x_, path_posets_y_;

  vector<HalfCycle> half_cycles_cx_, half_cycles_cy_;
  vector<vector<vector<HalfCycleVar>>> best_choice_hcycles_x_;
  vector<vector<vector<HalfCycleVar>>> best_choice_hcycles_y_;

  unordered_map<int64_t, int> MH[2];  // Set of edges mask which are currently fixed in bruteforce as best in row/col
  unordered_map<int64_t, vector<int>> cords_by_edges_mask[2];  // Set of complimentary coords for row/cols
  vector<int> MH_cov[2];  // Number of MH strategies for player X in row/column Y is MH_cov[X][Y]
  int MH_cov_max[2];      // Length of row/col for player X is MH_cov_max[X]

  vector<vector<int>> num_of_fails_by_cell_;
  vector<vector<int>> row_classes_;
  vector<vector<int>> col_classes_;

  vector<int> turns_;  // each value is in [-1; num_of_players), where -1 denotes terminal vertex
  vector<vector<Edge>> edges_;
  size_t start_vertex_;
  size_t num_of_players_;
  size_t num_of_edges_;
  vector<vector<vector<int>>> all_possible_players_strategies_;
  bool is_complete_;  // are costs of edges added?
  double ineq_sat_percentage_;
  LPSolver best_solver_first_player_;            // used only for SolveTwoPlayersRec
  LPSolver best_solver_second_player_;           // used only for SolveTwoPlayersRec
  vector<vector<int>> best_cells_cover_matrix_;  // used only for SolveTwoPlayersRec

  vector<int> CalcPlayersTotalSums(const vector<size_t>& strategy, size_t vertex_to_start, bool should_skip_visited) {
    size_t n = edges_.size();
    vector<int> total_costs(num_of_players_, 0);
    size_t curv = vertex_to_start;
    vector<int> is_vertex_visited(n);
    while (!is_vertex_visited[curv] && turns_[curv] != -1) {  // while not cycle and not terminal
      is_vertex_visited[curv] = 1;
      size_t index_of_edge_to_use = strategy[curv];
      assert(index_of_edge_to_use < edges_[curv].size());
      size_t nextv = edges_[curv][index_of_edge_to_use].finish;
      const vector<int>& edge_costs = edges_[curv][index_of_edge_to_use].cost;
      AddEdgeCosts(edge_costs, &total_costs);
      curv = nextv;
      assert(curv < n);
    }
    if (is_vertex_visited[curv]) {  // cycle
      if (should_skip_visited) {
        return total_costs;
      }
      vector<int> cycle_costs = CalcPlayersTotalSums(strategy, curv, true);
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        if (cycle_costs[player_idx] < 0) {
          cycle_costs[player_idx] = MinusInf();
        } else if (cycle_costs[player_idx] > 0) {
          cycle_costs[player_idx] = PlusInf();
        }
      }
      return cycle_costs;
    }
    return total_costs;
  }
};

struct GraphId {
  int cycle_size;
  int path_size;
  size_t build_path_choice_idx;
  size_t connect_cycle_choice_idx;
};

void AddEdge(int v, int u, vector<pair<int, int>>* edges) {
  edges->emplace_back(v, u);
}

int num_of_bits(int x) {
  return __builtin_popcount(x);
}

void GenAllPossibleChoicesForMasksRec(const vector<int>& limits,
                                      const vector<pair<int, int>>& bounds,
                                      vector<int>* cur_cortege,
                                      vector<vector<int>>* res) {
  size_t idx = cur_cortege->size();
  if (idx == limits.size()) {
    res->emplace_back(*cur_cortege);
    return;
  }
  for (size_t mask = 0; mask < limits[idx]; ++mask) {
    int cur_num_of_bits = num_of_bits(mask);
    if (cur_num_of_bits >= bounds[idx].first && cur_num_of_bits <= bounds[idx].second) {
      cur_cortege->emplace_back(mask);
      GenAllPossibleChoicesForMasksRec(limits, bounds, cur_cortege, res);
      cur_cortege->pop_back();
    }
  }
}

vector<vector<int>> GenAllPossibleChoicesForMasks(const vector<int>& limits, const vector<pair<int, int>>& bounds) {
  vector<vector<int>> res;
  vector<int> tmp;
  GenAllPossibleChoicesForMasksRec(limits, bounds, &tmp, &res);
  return res;
}

bool AreSpecialNashDigraphIsomorphic(const NashDigraph& n1, const NashDigraph& n2, const GraphId& graph_id) {
  int cycle_size = graph_id.cycle_size;
  int path_size = graph_id.path_size;
  vector<vector<int>> adj_m1 = n1.GetAdjacentMatrix(), adj_m2 = n2.GetAdjacentMatrix();
  vector<int> turns_n1 = n1.GetTurns();
  vector<int> turns_n2 = n2.GetTurns();
  int n = adj_m1.size();
  vector<size_t> path_perm(path_size);
  iota(path_perm.begin(), path_perm.end(), cycle_size + 1);
  vector<size_t> big_perm(n);
  vector<size_t> cycle_perm(cycle_size);
  iota(cycle_perm.begin(), cycle_perm.end(), 1);
  do {
    for (int i = 1; i <= cycle_size; ++i) {
      big_perm[i] = cycle_perm[i - 1];
    }
    do {
      for (int curv = cycle_size + 1; curv <= cycle_size + path_size; ++curv) {
        big_perm[curv] = path_perm[curv - cycle_size - 1];
      }
      bool is_cert_found = true;
      for (int curv = 0; curv < n; ++curv) {
        is_cert_found &= (turns_n1[curv] == turns_n2[big_perm[curv]]);
      }
      for (int v = 0; v < n; ++v) {
        for (int u = 0; u < n; ++u) {
          if (adj_m1[v][u]) {
            is_cert_found &= adj_m2[big_perm[v]][big_perm[u]];
          } else {
            is_cert_found &= !adj_m2[big_perm[v]][big_perm[u]];
          }
          if (!is_cert_found) {
            break;
          }
        }
        if (!is_cert_found) {
          break;
        }
      }
      if (is_cert_found) {
        return true;
      }
    } while (next_permutation(path_perm.begin(), path_perm.end()));
  } while (next_permutation(cycle_perm.begin(), cycle_perm.end()));
  return false;
}

bool BuildNashDigraphByGraphId(const GraphId& graph_id,
                               const vector<vector<int>>& choices_to_build_path,
                               const vector<vector<int>>& choices_to_connect_with_cycle,
                               NashDigraph* res) {
  int cycle_size = graph_id.cycle_size;
  int path_size = graph_id.path_size;
  size_t build_path_choice_idx = graph_id.build_path_choice_idx;
  size_t cycle_choice_idx = graph_id.connect_cycle_choice_idx;
  const vector<int>& choice_to_build_path = choices_to_build_path[build_path_choice_idx];
  const vector<int>& choice_to_connect_with_cycle = choices_to_connect_with_cycle[cycle_choice_idx];
  int n = 1 + cycle_size + path_size;
  vector<int> turns(n);
  turns[0] = -1;
  for (int i = 1; i <= cycle_size; ++i) {
    turns[i] = i % 2;
  }
  for (int vertex_idx = cycle_size + 1; vertex_idx < n; ++vertex_idx) {
    turns[vertex_idx] = -1;
  }
  turns[cycle_size + 1] = 0;  // player num
  vector<pair<int, int>> edges;
  // Edges on path
  bool is_bipartite = true;
  for (int vertex_in_path = 0; vertex_in_path < path_size; ++vertex_in_path) {
    int nghbr_mask = choice_to_build_path[vertex_in_path];
    /*
    if (vertex_in_path <= 2) {  // a -> b, b -> e, e -> f prefix
      int bit_pos = path_size - (vertex_in_path + 1) - 1;
      int next_bit = (nghbr_mask >> bit_pos);
      if (!next_bit) {
        return false;
      }
    }
    */

    /*
    if (vertex_in_path == 1) {  // a -> b, a -> e case
      int bit_pos = path_size - (vertex_in_path + 1) - 1;
      int next_bit = (nghbr_mask >> bit_pos);
      if (next_bit) {
        return false;
      }
    }
    */

    for (int next_vertex_num = vertex_in_path + 1; next_vertex_num < path_size; ++next_vertex_num) {
      int bit_pos = path_size - next_vertex_num - 1;
      int is_connected = (nghbr_mask >> bit_pos) & 1;
      if (is_connected) {
        int nxt_clr = (turns[cycle_size + 1 + vertex_in_path] ^ 1);
        int cur_clr = turns[cycle_size + 1 + next_vertex_num];
        if (cur_clr != -1 && nxt_clr != turns[cycle_size + 1 + next_vertex_num]) {
          is_bipartite = false;
        }
        AddEdge(cycle_size + 1 + vertex_in_path, cycle_size + 1 + next_vertex_num, &edges);
        turns[cycle_size + 1 + next_vertex_num] = nxt_clr;
      }
    }
    if (vertex_in_path != 0) {
      // TERMINALS
      AddEdge(cycle_size + 1 + vertex_in_path, 0, &edges);  // edge to terminal from prefix vertex except start
    }
  }
  if (!is_bipartite) {
    return false;
  }
  // Edges from path to cycle
  int num_of_outs = 0;
  vector<int> set_of_cycle_outs(cycle_size, 0);
  for (int vertex_in_path = 0; vertex_in_path < path_size; ++vertex_in_path) {
    int cycle_mask = choice_to_connect_with_cycle[vertex_in_path];
    for (int vertex_in_cycle = 0; vertex_in_cycle < cycle_size; ++vertex_in_cycle) {
      int is_connected = (cycle_mask >> vertex_in_cycle) & 1;
      num_of_outs += is_connected;
      if (is_connected) {
        set_of_cycle_outs[vertex_in_cycle]++;
        AddEdge(cycle_size + 1 + vertex_in_path, vertex_in_cycle + 1, &edges);
      }
    }
  }
  if (num_of_outs > 4) {
    return false;
  }
  for (int cycle_vertex = 0; cycle_vertex < cycle_size; ++cycle_vertex) {
    if (set_of_cycle_outs[cycle_vertex] > 1) {
      return false;
    }
  }

  /*
  if (num_of_intersections_on_cycle_outs == 0 && __builtin_popcount(choice_to_connect_with_cycle[0]) <= 2 &&
      __builtin_popcount(choice_to_connect_with_cycle[1] <= 2)) {
    cout << "Got already considered graph. Continue ...";
    return false;
  }
  */

  // Edges on cycle

  // New kind of cycle
  for (int vertex_in_cycle = 0; vertex_in_cycle < cycle_size; ++vertex_in_cycle) {
    int next_vertex_in_cycle = (vertex_in_cycle + 1) % cycle_size;
    AddEdge(vertex_in_cycle + 1, next_vertex_in_cycle + 1, &edges);
    AddEdge(vertex_in_cycle + 1, 0, &edges);
    /*
    if (vertex_in_cycle != 3) {
      AddEdge(vertex_in_cycle + 1, 0, &edges);
    }
    */
  }

  // AddEdge(4, 2, &edges);

  *res = NashDigraph(turns, 2, cycle_size + 1);

  for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    res->AddEmptyEdge(edges[edge_idx].first, edges[edge_idx].second, edge_idx);
  }

  // res->Print(false);
  if (!res->AreAllVerticesAccessibleFromStart() || !res->HasOnlyOneTerminal() ||
      !res->AreAllVerticesHasAtLeastTwoMoves()) {
    return false;
  }
  return true;
}

struct ThreadQueue {
  std::mutex m;
  std::condition_variable cv;
  std::queue<int> q;

  void PushRes(int x) {
    std::unique_lock<std::mutex> lock(m);
    q.push(x);
  }

  void Notify() {
    cv.notify_one();
  }

  void Pop(int* x) {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&] { return !q.empty(); });
    *x = q.front();
    q.pop();
  }
};

void CheckNashDigraphSample(int some_id,
                            const SolverParameters& solver_params,
                            size_t job_id,
                            NashDigraph G,
                            double* max_ineq_rate,
                            ThreadQueue* tq) {
  {
    std::unique_lock<std::mutex> lock(*solver_params.log_mutex);
    cout << "Starting job ID: " << job_id << endl;
    G.Print(false);
  }
  bool g_res;
  G.Preprocess(solver_params);
  G.CalcImprovementsTable(solver_params);
  g_res = G.SolveTwoPlayersCosts(solver_params);
  // G->CheckCorrectness();
  double cur_ineq_sat_percentage = G.GetIneqSatPercentage();
  *max_ineq_rate = max(*max_ineq_rate, cur_ineq_sat_percentage);
  {
    std::unique_lock<std::mutex> lock(*solver_params.log_mutex);
    cout << "Job ID: " << job_id << " is done" << endl;
  }
  tq->PushRes(g_res);
  tq->Notify();
  return;
}

bool TryToSolve(const SolverParameters& solver_params) {
  ifstream in(solver_params.offset_filename);
  assert(in.is_open());
  int offset = 0;
  in >> offset;
  in.close();
  vector<pair<NashDigraph, int>> digraphs;

  const size_t kDumpProgressPeriod = 10;

  double max_ineq_sat_percentage = 0.0;
  size_t best_graph_id = 0;
  vector<GraphId> graph_ids_to_check;
  vector<vector<int>> choices_to_connect_with_cycle;
  vector<vector<int>> choices_to_build_path;

  double max_ineq_rate = 0.0;

  map<vector<int>, vector<NashDigraph>> buckets;
  int total_num_of_classes = 0;
  int total_num_of_graphs = 0;

  vector<NashDigraph> graphs_to_check;

  const int kMaxTotalNumOfClasses = 10000;

  for (int path_size = solver_params.left_path_len_bound; path_size <= solver_params.right_path_len_bound;
       ++path_size) {
    cerr << "Start generating something new" << endl;
    vector<int> path_to_cycle_edges_ways_limits(path_size, (1 << solver_params.cycle_size));
    choices_to_connect_with_cycle =
      GenAllPossibleChoicesForMasks(path_to_cycle_edges_ways_limits, solver_params.num_of_edges_to_cycle_bounds);

    cerr << "All possible cycle outs are generated" << endl;

    vector<int> path_to_path_edges_ways_limits(path_size);
    for (int vertex_in_path = 1; vertex_in_path <= path_size; ++vertex_in_path) {
      int num_of_vertices_at_right = path_size - vertex_in_path;
      path_to_path_edges_ways_limits[vertex_in_path - 1] = (1 << num_of_vertices_at_right);
    }
    choices_to_build_path = GenAllPossibleChoices(path_to_path_edges_ways_limits);
    cerr << "Corteges are generated" << endl;

    for (size_t build_path_choice_idx = 0; build_path_choice_idx < choices_to_build_path.size();
         ++build_path_choice_idx) {
      if (total_num_of_classes > kMaxTotalNumOfClasses) {
        break;
      }
      for (size_t cycle_choice_idx = 0; cycle_choice_idx < choices_to_connect_with_cycle.size(); ++cycle_choice_idx) {
        if (total_num_of_classes > kMaxTotalNumOfClasses) {
          break;
        }
        GraphId cur_graph_id{solver_params.cycle_size, path_size, build_path_choice_idx, cycle_choice_idx};
        NashDigraph G;
        bool should_use =
          BuildNashDigraphByGraphId(cur_graph_id, choices_to_build_path, choices_to_connect_with_cycle, &G);
        if (!should_use) {
          continue;
        }
        total_num_of_graphs++;
        vector<NashDigraph>& cur_bucket = buckets[G.GetPassport()];
        bool is_same_class_found = false;
        for (const NashDigraph& lhs_dg : cur_bucket) {
          if (AreSpecialNashDigraphIsomorphic(lhs_dg, G, cur_graph_id)) {
            is_same_class_found = true;
            break;
          }
        }
        if (!is_same_class_found) {
          total_num_of_classes++;
          cout << "Graph id to check: " << total_num_of_classes << endl;
          cur_bucket.emplace_back(G);
          graphs_to_check.emplace_back(G);
        }
      }
    }
  }

  size_t num_of_threads = std::thread::hardware_concurrency();
  cerr << "Num of threads: " << num_of_threads << endl;

  cerr << "Total num of graphs: " << total_num_of_graphs << endl;

  ctpl::thread_pool pool(num_of_threads);

  vector<future<void>> jobs;

  ThreadQueue tq;

  ifstream done_in("done_ids.log");
  assert(done_in.is_open());

  int x;
  set<int> ready_ids;
  while (done_in >> x) {
    ready_ids.insert(x);
  }

  for (size_t job_id = 0; job_id < graphs_to_check.size(); ++job_id) {
    if (ready_ids.count(job_id) > 0) {
      continue;
    }
    jobs.emplace_back(pool.push(CheckNashDigraphSample,
                                std::cref(solver_params),
                                job_id,
                                std::move(graphs_to_check[job_id]),
                                &max_ineq_rate,
                                &tq));
  }

  cerr << "Total num of graphs: " << total_num_of_graphs << endl;

  for (size_t i = 0; i < graphs_to_check.size(); ++i) {
    int res;
    tq.Pop(&res);
    if (res) {
      std::cout << "YES!" << std::endl;
      pool.stop(false);
      return true;
    }
  }

  return false;
}

bool CheckIfTreeValidForTest(const vector<vector<int>>& edges, const vector<int>& parent, const vector<int>& turns) {
  int n = turns.size();
  int num_of_even_pos = 0;
  int num_of_odd_pos = 0;
  for (int x : turns) {
    if (x == 0) {
      num_of_even_pos++;
    } else if (x == 1) {
      num_of_odd_pos++;
    }
  }
  if (num_of_even_pos < 2 || num_of_odd_pos < 2) {
    cout << "Invalid number of positions condition" << endl;
    return false;  // we want at least 2 position for both players
  }
  int num_of_subtrees[2] = {0, 0};
  for (int v = 1; v < n; ++v) {
    if (turns[v] == -1) {
      continue;
    }
    int curv = parent[v];
    bool is_subtree_root = true;
    while (curv != 0) {
      if (turns[curv] == turns[v]) {
        is_subtree_root = false;
        break;
      }
      curv = parent[curv];
    }
    num_of_subtrees[turns[v]] += is_subtree_root;
  }
  if (num_of_subtrees[0] < 2 || num_of_subtrees[1] < 2) {
    cout << "Invalid pathways condition" << endl;
    return false;  // We want at least two vertices for both players on different pathways from root
  }
  return true;
}

void CheckTreeTests() {
  int kMaxTreeSize = 10;
  int kMinTreeSize = 5;
  int kNumOfIters = 10;

  for (int depth = 2; depth <= 8; ++depth) {
    int n = (1 << (depth + 1)) - 1;
    vector<int> turns(n, -1);
    turns[0] = 0;
    vector<pair<int, int>> edges;
    for (int v = 1; v < (1 << depth); ++v) {
      edges.emplace_back(v - 1, (v << 1) - 1);
      edges.emplace_back(v - 1, (v << 1));

      if (v < (1 << (depth - 1))) {
        turns[(v << 1) - 1] = (turns[v - 1] ^ 1);
        turns[(v << 1)] = (turns[v - 1] ^ 1);
      }
    }
    NashDigraph G(turns, 2, 0);
    for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
      G.AddEmptyEdge(edges[edge_idx].first, edges[edge_idx].second, edge_idx);
    }
    // G.Print(false);
    auto solver_params = SolverParameters{.are_pay_costs_positive = true,
                                          .is_special_six_cycle_len_graph = false,
                                          .left_path_len_bound = 2,
                                          .right_path_len_bound = 3,
                                          .cycle_size = 6,
                                          .num_of_edges_to_cycle_bounds = {{0, 2}, {0, 2}, {0, 2}},
                                          .offset_filename = "offset.txt",
                                          .should_shuffle_graphs = true,
                                          .need_to_remove_one_strategy = true};
    G.Preprocess(solver_params);
    G.CalcImprovementsTable(solver_params);
    // G.Print(false);
    assert(G.SolveTwoPlayersCosts(solver_params));
  }
  /*
  for (int it = 0; it < kNumOfIters; ++it) {
    int tree_size = GetRandomInt(kMinTreeSize, kMaxTreeSize);
    vector<int> parent(tree_size);
    vector<int> turns(tree_size);
    // Logariphmic tree size
    for (int i = 1; i < tree_size; ++i) {
      parent[i] = GetRandomInt(0, i - 1);
    }
    vector<vector<int>> edges_by_vertex(tree_size);
    for (int i = 1; i < tree_size; ++i) {
      edges_by_vertex[parent[i]].emplace_back(i);
    }
    queue<int> q;
    q.push(0);
    vector<pair<int, int>> edges;
    while (!q.empty()) {
      int v = q.front();
      q.pop();
      for (int u : edges_by_vertex[v]) {
        edges.emplace_back(make_pair(v, u));
        turns[u] = turns[v] ^ 1;
        q.push(u);
      }
      if (edges_by_vertex[v].empty()) {
        turns[v] = -1;
      }
    }
    int n = tree_size;
    for (int v = 1; v < tree_size; ++v) {
      if (turns[v] == -1) {
        continue;
      }
      if (edges_by_vertex[v].size() < 2) {
        parent.resize(n + 1);  // n^2 totally, doesn't matter
        parent[n] = v;
        turns.resize(n + 1);
        turns[n] = -1;
        edges_by_vertex.resize(n + 1);
        edges_by_vertex[v].emplace_back(n);
        edges.emplace_back(make_pair(v, n));
        n++;
      }
    }
    cout << "Tree size: " << tree_size << " " << n << endl;
    tree_size = n;
    if (!CheckIfTreeValidForTest(edges_by_vertex, parent, turns)) {
      cout << "Got invalid tree for tests, continue ..." << endl;
      continue;
    }
    NashDigraph G(turns, 2, 0);
    for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
      G.AddEmptyEdge(edges[edge_idx].first, edges[edge_idx].second, edge_idx);
    }
    // G.Print(false);
    auto solver_params = SolverParameters{.are_pay_costs_positive = true,
                                          .is_special_six_cycle_len_graph = false,
                                          .left_path_len_bound = 2,
                                          .right_path_len_bound = 3,
                                          .cycle_size = 6,
                                          .num_of_edges_to_cycle_bounds = {{0, 2}, {0, 2}, {0, 2}},
                                          .offset_filename = "offset.txt",
                                          .should_shuffle_graphs = true,
                                          .need_to_remove_one_strategy = true};
    G.Preprocess(solver_params);
    G.CalcImprovementsTable(solver_params);
    G.Print(false);
    assert(G.SolveTwoPlayersCosts(solver_params));
  }
  */
}

void CheckNegativeCostsTests() {
  vector<int> turns = {0, 1, 1, -1};
  NashDigraph G(turns, 2, 0);
  G.AddEmptyEdge(0, 1, 0);
  G.AddEmptyEdge(1, 2, 1);
  G.AddEmptyEdge(2, 0, 2);
  G.AddEmptyEdge(0, 2, 3);
  G.AddEmptyEdge(1, 0, 4);
  G.AddEmptyEdge(2, 1, 5);
  G.AddEmptyEdge(2, 3, 6);
  G.Print(false);
  auto solver_params = SolverParameters{.are_pay_costs_positive = false,
                                        .is_special_six_cycle_len_graph = false,
                                        .left_path_len_bound = 2,
                                        .right_path_len_bound = 3,
                                        .cycle_size = 6,
                                        .num_of_edges_to_cycle_bounds = {{0, 2}, {0, 2}, {0, 2}},
                                        .offset_filename = "offset.txt",
                                        .should_shuffle_graphs = true,
                                        .need_to_remove_one_strategy = false};
  G.Preprocess(solver_params);
  G.CalcImprovementsTable(solver_params);
  assert(G.SolveTwoPlayersCosts(solver_params));
  G.CheckCorrectness();
}

void TestIsomoprhicChecker() {
  int cycle_size = 3;
  int path_size = 3;
  GraphId graph_id{.cycle_size = cycle_size, .path_size = path_size};
  int n = cycle_size + path_size + 1;
  vector<pair<int, int>> edges1, edges2;
  vector<int> turns(n);
  for (int v = 0; v < cycle_size; ++v) {
    edges1.emplace_back(v + 1, (v + 1) % cycle_size + 1);
    edges1.emplace_back(v + 1, 0);
  }
  edges2 = edges1;
  edges1.emplace_back(4, 5);
  edges1.emplace_back(4, 6);
  edges1.emplace_back(5, 1);
  edges1.emplace_back(6, 0);

  edges2.emplace_back(4, 5);
  edges2.emplace_back(4, 6);
  edges2.emplace_back(6, 1);
  edges2.emplace_back(5, 0);
  NashDigraph G1(turns, 2, cycle_size + 1), G2(turns, 2, cycle_size + 1);
  int edge_idx = 0;
  for (const auto& edge : edges1) {
    G1.AddEmptyEdge(edge.first, edge.second, edge_idx);
    edge_idx++;
  }
  edge_idx = 0;
  for (const auto& edge : edges2) {
    G2.AddEmptyEdge(edge.first, edge.second, edge_idx);
    edge_idx++;
  }
  assert(AreSpecialNashDigraphIsomorphic(G1, G2, graph_id));
}

int main() {
  LPSolver::LaunchPython();

  // CheckTreeTests();
  // CheckNegativeCostsTests();
  // CheckTreeTests();
  // NashDigraph G("input.txt", false);

  /*
  cerr << G.CountNumOfNE() << endl;
  */
  /*
  auto f1 = G.CheckHypo(0);
  auto f2 = G.CheckHypo(1);
  for (auto P : f1) {
    cout << P.first << " " << P.second << endl;
  }
  cout << "BREAK!" << endl;
  for (auto P : f2) {
    cout << P.first << " " << P.second << endl;
  }
  */
  // freopen("input.txt", "r", stdin);
  /*
  int it = 0;
  while (true) {
    it++;
    NashDigraph G("input.txt", true);
    if (it % 100 == 0) {
      cerr << it << endl;
    }
    */
  /*
 NashDigraph G("input.txt", true);
 auto f1 = G.CheckHypo(0);
 auto f2 = G.CheckHypo(1);

 cout << IsIntersection(f1, f2) << endl;
 */

  // G.CountNumOfNE();

  // cout << G.CountNumOfNE() << endl;
  // cout << G.SolveThreePlayersCosts() << endl;
  // G.CheckCorrectnessThree();

  // cout << G.AreAllVerticesAccessibleFromStart() << endl;
  // cout << G.SolveTwoPlayersCosts(true) << endl;
  // G.CheckCorrectness();
  // cout << G.GetIneqSatPercentage() << endl;
  // cout << G.CountNumOfNE() << endl;
  // CheckTreeTests();
  // CheckNegativeCostsTests();

  /*
  NashDigraph G("input.txt", false);

  std::mutex log_mutex;

  auto solver_params = SolverParameters{.are_pay_costs_positive = true,
                                        .is_special_six_cycle_len_graph = false,
                                        .left_path_len_bound = 2,
                                        .right_path_len_bound = 2,
                                        .cycle_size = 8,
                                        .num_of_edges_to_cycle_bounds = {{1, 6}, {0, 3}, {0, 3}, {0, 3}},
                                        .offset_filename = "offset.txt",
                                        .should_shuffle_graphs = true,
                                        .need_to_remove_one_strategy = false,
                                        .log_mutex = &log_mutex};

  G.Preprocess(solver_params);
  G.CalcImprovementsTable(solver_params);
  cout << G.SolveTwoPlayersCosts(solver_params) << endl;
  */

  std::mutex log_mutex;
  bool res = TryToSolve(SolverParameters{.are_pay_costs_positive = true,
                                         .is_special_six_cycle_len_graph = false,
                                         .left_path_len_bound = 2,
                                         .right_path_len_bound = 2,
                                         .cycle_size = 8,
                                         .num_of_edges_to_cycle_bounds = {{1, 6}, {0, 3}, {0, 3}, {0, 3}},
                                         .offset_filename = "offset.txt",
                                         .should_shuffle_graphs = true,
                                         .need_to_remove_one_strategy = false,
                                         .log_mutex = &log_mutex});
  if (res) {
    cout << "VICTORY!" << endl;
  }

  // TestIsomoprhicChecker();

  // cout << TryToSolve(2, 3, 4, {{0, 2}, {0, 2}, {0, 0}, {0, 2}, {0, 0}}, "offset.txt", true) << endl; //offset -
  // 1732
  // // 0.991 930
  // 2250 - for cycle_size = 3
  // 320 for {3, 3, 3} and cycle_size = 6
  // cout << TryToSolve(2, 3, 6, 3, 208, true);

  LPSolver::ReleasePython();
  return 0;
}