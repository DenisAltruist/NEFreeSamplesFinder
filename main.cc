// -I/usr/include/python2.7/ -lpython2.7

#include <bits/stdc++.h>
#include <Python.h>

using namespace std;

mt19937 mt(12345);

class TimeCounter {
 public:
  enum TimeUnits {
    kMicrosecs,
    kMillisecs,
    kSeconds
  };
 private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  TimeUnits time_unit_;
  bool running;
  static const auto now() {
    return std::chrono::steady_clock::now();
  }
 public:
  TimeCounter() : time_unit_(kMillisecs), running(false) {};
  explicit TimeCounter(TimeUnits timeUnit) : time_unit_(timeUnit), running(false) {};
  void start() {
    start_time_ = now();
    running = true;
  }
  double finish() {
    if (!running) {
      assert(0);
    }
    running = false;
    const std::chrono::time_point<std::chrono::steady_clock> endTime = now();
    switch (time_unit_) {
      case kMicrosecs: {
        const std::chrono::duration<double, std::micro> elapsedTime = endTime - start_time_;
        return elapsedTime.count();
      }
      case kMillisecs: {
        const std::chrono::duration<double, std::milli> elapsedTime = endTime - start_time_;
        return elapsedTime.count();
      }
      case kSeconds: {
        const std::chrono::duration<double, std::ratio<1>> elapsedTime = endTime - start_time_;
        return elapsedTime.count();
      }
      default: {
        assert(0);
      }
    }
  }
};

int PlusInf() {
  return numeric_limits<int>::max();
}

int MinusInf() {
  return numeric_limits<int>::lowest();
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
  int finish;
  vector<int> cost;
  int idx;
};

class LPSolver {
  public:
    static void LaunchPython() {
      Py_Initialize();

      PyRun_SimpleString("import sys");
      PyRun_SimpleString("import os");
      PyRun_SimpleString("sys.path.append(os.getcwd())");

      PyObject* p_name = PyString_FromString("lp_solver");
      CheckPyObjectFailure(p_name);

      PyObject* p_module = PyImport_Import(p_name);
      CheckPyObjectFailure(p_module);

      PyObject* p_dict = PyModule_GetDict(p_module);
      CheckPyObjectFailure(p_dict);

      PyObject* p_func = PyDict_GetItemString(p_dict, "is_feasible_positive");
      CheckPyObjectFailure(p_func);

      feasibility_func_ = p_func;
    }

    LPSolver(int num_of_variables) : num_of_variables_(num_of_variables) {}

    void PushInequality(const vector<int>& coeffs) { // last coeff is number after comparison sing. a_1x_1 + ... + a_nx_n <= c
      assert(num_of_variables_ + 1 == coeffs.size());
      inequalities_.emplace_back(coeffs);
    }

    void PopInequality() {
      assert(!inequalities_.empty());
      inequalities_.pop_back();
    }

    size_t Size() {
      return inequalities_.size();
    }

    bool IsFeasible() {
      // cout << "Feasible check for " << inequalities_.size() <<  " inequalities" << endl;
      if (inequalities_.empty()) {
        return true;
      }
      TimeCounter feasibility_check_timer(TimeCounter::kSeconds);
      feasibility_check_timer.start(); // 0.1 for check with files and shell call.
      PyObject* p_args = PyTuple_New(2);
      vector<int> bounds(inequalities_.size());

      cout << inequalities_.size() * num_of_variables_ << endl;

      for (size_t ineq_idx = 0; ineq_idx < inequalities_.size(); ++ineq_idx) {
        bounds[ineq_idx] = inequalities_[ineq_idx][num_of_variables_];
        inequalities_[ineq_idx].resize(num_of_variables_ - 1);
      }

      PyObject* call_result = PyObject_CallObject(feasibility_func_, p_args);
      CheckPyObjectFailure(call_result);

      int res = PyInt_AsLong(call_result);
      cout << feasibility_check_timer.finish() << endl;

      for (size_t ineq_idx = 0; ineq_idx < inequalities_.size(); ++ineq_idx) {
        inequalities_[ineq_idx].resize(num_of_variables_);
      }

      return res;
    }

  private:
    static PyObject* feasibility_func_;

    PyObject* ineq_matrix_;
    PyObject* bounds_;

    static void CheckPyObjectFailure(PyObject* object) {
      if (object == nullptr) {
        PyErr_Print();
        exit(1);
      }
    }

    PyObject* PListFromVector(const vector<int>& nums) {
      PyObject* l = PyList_New(nums.size());
      for (size_t i = 0; i < nums.size(); ++i) {
        PyList_SET_ITEM(l, i, PyInt_FromLong(nums[i]));
      }
      return l;
    }

    PyObject* PMatrixFromVector(const vector<vector<int>>& matrix) {
      PyObject* l = PyList_New(matrix.size());
      for (size_t i = 0; i < matrix.size(); ++i) {
        PyList_SET_ITEM(l, i, PListFromVector(matrix[i]));
      }
      return l;
    }

    int num_of_variables_;
    vector<vector<int>> inequalities_; // the form of each inequality: a_1x_1 + a_2x_2 + ... + a_nx_n >= 0
};

PyObject* LPSolver::feasibility_func_ = nullptr;

class NashDigraph {
  public:
    NashDigraph(const string& path_to_file, bool is_complete) {
      is_complete_ = is_complete;
      ifstream in(path_to_file);
      assert(in.is_open());
      int num_of_vertices, num_of_edges, num_of_players;
      in >> num_of_vertices >> num_of_edges >> num_of_players >> start_vertex_;
      num_of_edges_ = num_of_edges;
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
        if (is_complete) { // otherwise we will add (0, 0, ..., 0) costs
          for (int cost_idx = 0; cost_idx < num_of_players; ++cost_idx) {
            in >> edge_cost[cost_idx];
          }
        }
        AddEdge(v, u, edge_cost, edge_idx);
      }
      num_of_players_ = num_of_players;
      Preprocess();
    }

    // For this constructor edges should be added manually
    NashDigraph(const vector<int>& turns, int num_of_players, size_t start_vertex) : 
      turns_(turns),
      edges_(vector<vector<Edge>>(turns.size())),
      start_vertex_(start_vertex),
      num_of_players_(num_of_players),
      num_of_edges_(0) {
    }

    void AdjustEmptyTurns() {
      for (size_t v = 0; v < turns_.size(); ++v) {
        int turn = turns_[v];
        if (turn == -1) {
          continue;
        }
        if (edges_[v].empty()) {
          turns_[v] = -1;
        }
      }
    }

    void Preprocess() {
      AdjustEmptyTurns();
      CalcAllPossiblePlayersStrategies();
    }

    void Print(bool with_costs) {
      cout << turns_.size() << " " << num_of_edges_ << " " << num_of_players_ << " " << start_vertex_ << endl;
      for (size_t turn_idx = 0; turn_idx < turns_.size(); ++turn_idx) {
        cout << turns_[turn_idx] << " ";
      }
      cout << endl;
      for (size_t v = 0; v < turns_.size(); ++v) {
        for (const Edge& e : edges_[v]) {
          cout << v << " " << e.finish;
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
      edges_[v].push_back(Edge{u, costs, edge_idx});
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
      vector<size_t> player_own_vertices;
      for (size_t v = 0; v < n; ++v) {
        if (turns_[v] == static_cast<int>(player_idx)) {
          player_num_of_edges_limits.emplace_back(edges_[v].size());
          player_own_vertices.emplace_back(v);
        }
      }
      all_possible_players_strategies_[player_idx] = GenAllPossibleChoices(player_num_of_edges_limits);
    }

    void CalcAllPossiblePlayersStrategies() {
      all_possible_players_strategies_.resize(num_of_players_);
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        CalcPlayerStrategies(player_idx);
      }
    }

    void ApplyPlayerStrategyToGlobalOne(
      const vector<int>& player_strategy, 
      size_t player_idx, 
      vector<size_t>* all_players_strategy
    ) {
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

    int CountNumOfNE() {
      size_t n = turns_.size();
      vector<int> num_of_strategies_limits(num_of_players_);
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        num_of_strategies_limits[player_idx] = all_possible_players_strategies_[player_idx].size();
      }
      vector<vector<int>> all_possible_strategies_corteges =  GenAllPossibleChoices(num_of_strategies_limits);
      int total_num_of_corteges = all_possible_strategies_corteges.size();
      int num_of_corteges_in_ne = 0;
      //cout << "Total num of tuples of strategies: " << total_num_of_corteges << endl;
      for (const vector<int>& strategy_cortege : all_possible_strategies_corteges) {
        vector<size_t> all_players_strategy(n); // 0 will remain for terminals
        for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
          size_t strategy_for_cur_player_to_use_idx = strategy_cortege[player_idx];
          const vector<int>& cur_player_strategy = all_possible_players_strategies_[player_idx][strategy_for_cur_player_to_use_idx];
          ApplyPlayerStrategyToGlobalOne(cur_player_strategy, player_idx, &all_players_strategy);
        }
        num_of_corteges_in_ne += IsStrategyNE(all_players_strategy);
      }
      //cout << "Num of corteges in NE: " << num_of_corteges_in_ne << endl;
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

    vector<vector<int>> GetLinFuncsForPlayersByGlobalStrategy(const vector<size_t>& all_players_strategy) {
      vector<vector<int>> used_edges_by_player_idx(num_of_players_);
      size_t curv = start_vertex_;
      vector<int> is_vertex_used(turns_.size());
      while (!is_vertex_used[curv] && turns_[curv] != -1) {
        is_vertex_used[curv] = 1;
        assert(curv < all_players_strategy.size());
        size_t index_of_edge_to_use = all_players_strategy[curv];
        assert(index_of_edge_to_use < edges_[curv].size());
        size_t nextv = edges_[curv][index_of_edge_to_use].finish;
        assert(turns_[curv] < used_edges_by_player_idx.size());
        used_edges_by_player_idx[turns_[curv]].emplace_back(edges_[curv][index_of_edge_to_use].idx); 
        curv = nextv;
        assert(curv < turns_.size());
      }
      if (is_vertex_used[curv]) { // got cycle
        return vector<vector<int>>(num_of_players_); 
      }
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        vector<int> res(num_of_edges_);
        for (size_t edge_idx : used_edges_by_player_idx[player_idx]) {
          assert(edge_idx < num_of_edges_);
          res[edge_idx] = 1;
        }
        used_edges_by_player_idx[player_idx] = res;
      }
      return used_edges_by_player_idx;
    }
    
    // returns true if improved by addition
    bool AddInequality(const vector<int>& old_func, const vector<int>& new_func, LPSolver* lp_solver) {
      if (old_func.empty()) {
        if (!new_func.empty()) {
          return true;
        }
        return false;
      }
      if (new_func.empty()) {
        return false;
      }
      vector<int> ineq(num_of_edges_);
      assert(old_func.size() == num_of_edges_);
      assert(new_func.size() == num_of_edges_);
      for (size_t var_idx = 0; var_idx < num_of_edges_; ++var_idx) {
        ineq[var_idx] = new_func[var_idx] - old_func[var_idx];
      }
      ineq.emplace_back(-1);
      lp_solver->PushInequality(ineq);
      return true;
    }

    bool SolveTwoPlayersPositiveCostsRec(
      const vector<vector<pair<vector<int>, vector<int>>>>& linear_funcs_by_cell,
      int cx, 
      int cy,
      vector<vector<int>>* is_cell_used, 
      LPSolver* lp_solver
    ) {
      int n = is_cell_used->size();
      int m = (*is_cell_used)[0].size();
      assert(cx < n);
      assert(cy < m);
      ineq_sat_percentage_ = max(ineq_sat_percentage_, double(lp_solver->Size()) / (n * m));
      if ((*is_cell_used)[cx][cy]) {
        for (int tx = 0; tx < n; ++tx) {
          for (int ty = 0; ty < m; ++ty) {
            if (!(*is_cell_used)[tx][ty]) {
              return SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, tx, ty, is_cell_used, lp_solver);
            }
          }
        }
        return true;
      }
      (*is_cell_used)[cx][cy] = 1;
      // finding cell to improve for the first player
      for (int tx = 0; tx < n; ++tx) {
        size_t old_lp_solver_size = lp_solver->Size();
        if (tx == cx) { 
          continue;
        }
        
        const vector<int>& best_linear_func = linear_funcs_by_cell[tx][cy].first;
        if (best_linear_func.empty()) {
          continue;
        }
        for (int func_idx = 0; func_idx < n; ++func_idx) {
          if (func_idx == tx) {
            continue;
          }
          AddInequality(linear_funcs_by_cell[func_idx][cy].first, linear_funcs_by_cell[tx][cy].first, lp_solver);
        }
        if (lp_solver->IsFeasible()) {
          vector<pair<int, int>> colored_cells;
          for (int wx = 0; wx < n; ++wx) {
            if (wx == tx) {
              continue;
            }
            if (!(*is_cell_used)[wx][cy]) {
              colored_cells.emplace_back(wx, cy);
              (*is_cell_used)[wx][cy] = 1;
            }
          }
          bool branch_result = SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, cx, cy, is_cell_used, lp_solver);
          if (branch_result) {
            return true;
          }
          for (const auto& colored_cell : colored_cells) {
            (*is_cell_used)[colored_cell.first][colored_cell.second] = 0;
          }
        }
      
        while (lp_solver->Size() != old_lp_solver_size) {
          lp_solver->PopInequality();
        } 
      }
      // finding cell to improve for the second player
      for (int ty = 0; ty < m; ++ty) {
         size_t old_lp_solver_size = lp_solver->Size();
        if (ty == cy) { 
          continue;
        }
        const vector<int>& best_linear_func = linear_funcs_by_cell[cx][ty].first;
        if (best_linear_func.empty()) {
          continue;
        }
        for (int func_idx = 0; func_idx < m; ++func_idx) {
          if (func_idx == ty) {
            continue;
          }
          AddInequality(linear_funcs_by_cell[cx][func_idx].first, linear_funcs_by_cell[cx][ty].first, lp_solver);
        }
        if (lp_solver->IsFeasible()) {
          vector<pair<int, int>> colored_cells;
          for (int wy = 0; wy < m; ++wy) {
            if (wy == ty) {
              continue;
            }
            if (!(*is_cell_used)[cx][wy]) {
              colored_cells.emplace_back(cx, wy);
              (*is_cell_used)[cx][wy] = 1;
            }
          }
          bool branch_result = SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, cx, cy, is_cell_used, lp_solver);
          if (branch_result) {
            return true;
          }
          for (const auto& colored_cell : colored_cells) {
            (*is_cell_used)[colored_cell.first][colored_cell.second] = 0;
          }
        }
        while (lp_solver->Size() != old_lp_solver_size) {
          lp_solver->PopInequality();
        } 
      }
      (*is_cell_used)[cx][cy] = 0;
      return false;
    }

    double GetIneqSatPercentage() {
      return ineq_sat_percentage_;
    }

    bool SolveTwoPlayersPositiveCosts() {
      assert(num_of_players_ == 2);
      int n = all_possible_players_strategies_[0].size();
      int m = all_possible_players_strategies_[1].size();
      ineq_sat_percentage_ = 0.0;
      cerr << "Num of strategies for players: " << n << " " << m << endl;
      if (n == 0 || m == 0) {
        return false;
      }
      vector<vector<int>> is_pair_of_strategies_used(n, vector<int>(m));
      vector<vector<pair<vector<int>, vector<int>>>> linear_funcs_by_cell(
        n, 
        vector<pair<vector<int>, vector<int>>>(m) // if vector is empty, then both vectors should be empty and this is a cycle
      );
      for (int cx = 0; cx < n; ++cx) {
        for (int cy = 0; cy < m; ++cy) {
          vector<size_t> all_players_strategy(turns_.size());
          ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][cx], 0, &all_players_strategy);
          ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][cy], 1, &all_players_strategy);
          vector<vector<int>> lin_funcs = GetLinFuncsForPlayersByGlobalStrategy(all_players_strategy);
          linear_funcs_by_cell[cx][cy] = make_pair(lin_funcs[0], lin_funcs[1]);
        }
      }
      LPSolver lp_solver(num_of_edges_); // num of edges in actually num of variables
      // Conditions x_i > 0 are already accounted in 'lp_solver.py'
      return SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, 0, 0, &is_pair_of_strategies_used, &lp_solver);
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

  private:
    vector<int> CalcPlayersTotalSums(const vector<size_t>& strategy, size_t vertex_to_start, bool should_skip_visited) {
      size_t n = edges_.size();
      vector<int> total_costs(num_of_players_, 0);
      size_t curv = vertex_to_start;
      vector<int> is_vertex_visited(n);
      while (!is_vertex_visited[curv] && turns_[curv] != -1) { // while not cycle and not terminal
        is_vertex_visited[curv] = 1;
        size_t index_of_edge_to_use = strategy[curv];
        assert(index_of_edge_to_use < edges_[curv].size());
        size_t nextv = edges_[curv][index_of_edge_to_use].finish;
        const vector<int>& edge_costs = edges_[curv][index_of_edge_to_use].cost;
        AddEdgeCosts(edge_costs, &total_costs);
        curv = nextv;
        assert(curv < n);
      }
      if (is_vertex_visited[curv]) { // cycle
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

    vector<int> turns_; // each value is in [-1; num_of_players), where -1 denotes terminal vertex
    vector<vector<Edge>> edges_;
    size_t start_vertex_;
    size_t num_of_players_;
    size_t num_of_edges_;
    vector<vector<vector<int>>> all_possible_players_strategies_;
    bool is_complete_; // are costs of edges added?
    double ineq_sat_percentage_;
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

bool TryToSolve(int ps_lb, int ps_rb, int cycle_size, int max_num_of_edges_from_path_to_cycle, int offset, bool should_shuffle) {
  double max_ineq_sat_percentage = 0.0;
  vector<GraphId> graph_ids_to_check;
  vector<vector<int>> choices_to_connect_with_cycle;
  vector<vector<int>> choices_to_build_path;
  for (int path_size = ps_lb; path_size <= ps_rb; ++path_size) {

    vector<int> path_to_cycle_edges_ways_limits(path_size, (1 << cycle_size));
    choices_to_connect_with_cycle = GenAllPossibleChoices(path_to_cycle_edges_ways_limits);
    vector<vector<int>> sifted_cycle_choices; // at most 'max_num_of_edges_from_path_to_cycle' edges
    for (const vector<int>& cycle_choice : choices_to_connect_with_cycle) {
      bool is_bad = false;
      for (int x : cycle_choice) {
        if (num_of_bits(x) > max_num_of_edges_from_path_to_cycle) {
          is_bad = true;
          break;
        }
      }
      if (!is_bad) {
        sifted_cycle_choices.emplace_back(cycle_choice);
      }
    }
    choices_to_connect_with_cycle = sifted_cycle_choices;

    vector<int> path_to_path_edges_ways_limits(path_size);
    for (int vertex_in_path = 1; vertex_in_path <= path_size; ++vertex_in_path) {
      int num_of_vertices_at_right = path_size - vertex_in_path;
      path_to_path_edges_ways_limits[vertex_in_path - 1] = (1 << num_of_vertices_at_right); 
    }
    choices_to_build_path = GenAllPossibleChoices(path_to_path_edges_ways_limits);

    for (size_t build_path_choice_idx = 0; build_path_choice_idx < choices_to_build_path.size(); ++build_path_choice_idx) {
      for (size_t cycle_choice_idx = 0; cycle_choice_idx < choices_to_connect_with_cycle.size(); ++cycle_choice_idx) {
        graph_ids_to_check.emplace_back(GraphId{cycle_size, path_size, build_path_choice_idx, cycle_choice_idx});
      }
    }
  }
  if (should_shuffle) {
    Shuffle(&graph_ids_to_check);
  }
  for (size_t graph_id_idx = offset; graph_id_idx < graph_ids_to_check.size(); ++graph_id_idx) {
    const auto& graph_id = graph_ids_to_check[graph_id_idx];
    int cycle_size = graph_id.cycle_size;
    int path_size = graph_id.path_size;
    size_t build_path_choice_idx = graph_id.build_path_choice_idx;
    size_t cycle_choice_idx = graph_id.connect_cycle_choice_idx;
    const vector<int>& choice_to_build_path = choices_to_build_path[build_path_choice_idx];
    const vector<int>& choice_to_connect_with_cycle = choices_to_connect_with_cycle[cycle_choice_idx];
    cout << "Check graph for graph id " << graph_id_idx << endl;
    int n = 1 + cycle_size + path_size;
    vector<int> turns(n);
    turns[0] = -1;
    for (int i = 1; i <= cycle_size; ++i) {
      turns[i] = i % 2;
    }
    for (int vertex_idx = cycle_size + 1; vertex_idx < n; ++vertex_idx) {
      turns[vertex_idx] = 0;
    }
    vector<pair<int, int>> edges;
    // Edges on path
    for (int vertex_in_path = 0; vertex_in_path < path_size; ++vertex_in_path) {
      int nghbr_mask = choice_to_build_path[vertex_in_path];
      for (int next_vertex_num = vertex_in_path + 1; next_vertex_num < path_size; ++next_vertex_num) {
        int bit_pos = path_size - next_vertex_num - 1;
        int is_connected = (nghbr_mask >> bit_pos) & 1;
        if (is_connected) {
          AddEdge(cycle_size + 1 + vertex_in_path, cycle_size + 1 + next_vertex_num, &edges);
          turns[cycle_size + 1 + next_vertex_num] = (turns[cycle_size + 1 + vertex_in_path] ^ 1);
        }
      }
    }
    // Edges from path to cycle
    for (int vertex_in_path = 0; vertex_in_path < path_size; ++vertex_in_path) {
      int cycle_mask = choice_to_connect_with_cycle[vertex_in_path];
      for (int vertex_in_cycle = 0; vertex_in_cycle < cycle_size; ++vertex_in_cycle) {
        int is_connected = (cycle_mask >> vertex_in_cycle) & 1;
        if (is_connected) {
          AddEdge(cycle_size + 1 + vertex_in_path, vertex_in_cycle + 1, &edges);
        }
      }
    }
    // Edges on cycle
    
    for (int vertex_in_cycle = 0; vertex_in_cycle < cycle_size; ++vertex_in_cycle) {
      int next_vertex_in_cycle = (vertex_in_cycle + 1) % cycle_size;
      AddEdge(vertex_in_cycle + 1, next_vertex_in_cycle + 1, &edges);
      AddEdge(vertex_in_cycle + 1, 0, &edges);
    }

    NashDigraph G(turns, 2, cycle_size + 1);
    for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
      G.AddEmptyEdge(edges[edge_idx].first, edges[edge_idx].second, edge_idx);
    }
    if (!G.AreAllVerticesAccessibleFromStart()) {
      cerr << "This graph will be skipped, as not all vertices are accessible from start" << endl;
      continue;
    }
    G.Print(false);
    G.Preprocess();
    bool g_res = G.SolveTwoPlayersPositiveCosts();
    max_ineq_sat_percentage = max(max_ineq_sat_percentage, G.GetIneqSatPercentage());
    cout << "Current max inequality saturation percentage: " << max_ineq_sat_percentage << endl;
    if (g_res) {
      return true;
    }
  }
  return false;
}

int main() {
    LPSolver::LaunchPython();
    //freopen("input.txt", "r", stdin);
    NashDigraph G("input.txt", false);
    // cout << G.AreAllVerticesAccessibleFromStart() << endl;
    cout << G.SolveTwoPlayersPositiveCosts() << endl;
    //cout << G.GetIneqSatPercentage() << endl;
    //cout << G.CountNumOfNE() << endl;
    /*
      2, 3, 6, 3, ..., true => offset = 75
    */
    //cout << TryToSolve(2, 3, 3, 3, 0, true);

    Py_Finalize();
    return 0;
}