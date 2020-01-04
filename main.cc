// g++ main.cc -O3 -o main -I/usr/include/python2.7 -lpython2.7

#include <bits/stdc++.h>
#include <Python.h>

using namespace std;

const int kEdgeCostLimit = 25;

mt19937 mt(123);

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

      PyObject* p_name = PyUnicode_FromString("lp_solver");
      CheckPyObjectFailure(p_name);

      PyObject* p_module = PyImport_Import(p_name);
      CheckPyObjectFailure(p_module);

      PyObject* p_dict = PyModule_GetDict(p_module);
      CheckPyObjectFailure(p_dict);

      feasibility_func_ = PyDict_GetItemString(p_dict, "is_feasible");
      CheckPyObjectFailure(feasibility_func_);

      sol_func_ = PyDict_GetItemString(p_dict, "solve");
      CheckPyObjectFailure(sol_func_);
    }

    static void ReleasePython() {
      Py_Finalize();
    }

    LPSolver() = default;

    LPSolver(int num_of_variables) : num_of_variables_(num_of_variables) {}

    pair<bool, vector<double>> GetSolution() {
      vector<double> res_list;
      /*
      const int kNumOfRandomAttempts = 1;
      for (int it = 0; it < kNumOfRandomAttempts; ++it) {
        vector<double> trial_sol(num_of_variables_);
        for (int var_idx = 0; var_idx < num_of_variables_; ++var_idx) {
          trial_sol[var_idx] = GetRandomInt(1, kEdgeCostLimit);
        }
        bool res = CheckSolution(trial_sol);
        if (res) {
          return make_pair(true, trial_sol);
        }
      }
      */


      PyObject* res = CallCvxopt(sol_func_);
      CheckPyObjectFailure(res);
      PyObject* first_res = PyTuple_GetItem(res, 0);
      CheckPyObjectFailure(first_res);
      PyObject* second_res = PyTuple_GetItem(res, 1);
      CheckPyObjectFailure(second_res);
      int sol_len = PyObject_Length(second_res);
      for (int i = 0; i < sol_len; ++i) {
        PyObject* item = PyList_GetItem(second_res, i);
        CheckPyObjectFailure(item);
        res_list.emplace_back(PyFloat_AsDouble(item));
      }
      return make_pair(PyLong_AsLong(first_res), res_list);
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

    bool PushInequality(const vector<int>& ineq, int bound) { // last coeff is number after comparison sing. a_1x_1 + ... + a_nx_n <= c
      ineqs_.emplace_back(ineq);
      bounds_.emplace_back(bound);
      return true;
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
    static PyObject* feasibility_func_;
    static PyObject* sol_func_;

    PyObject* CallCvxopt(PyObject* call_function) {
      //TimeCounter feasibility_check_timer(TimeCounter::kSeconds);
      //feasibility_check_timer.start(); // 0.1 for check with files and shell call.
      PyObject* p_args = PyTuple_New(2);

      assert(!ineqs_.empty());
      assert(!bounds_.empty());

      PyTuple_SetItem(p_args, 0, PMatrixFromVector(ineqs_, "int"));
      PyTuple_SetItem(p_args, 1, PListFromVector(bounds_, "int"));

      PyObject* call_result = PyObject_CallObject(call_function, p_args);

      Py_DECREF(p_args);

      CheckPyObjectFailure(call_result);
      return call_result;
    }

    static void CheckPyObjectFailure(PyObject* object) {
      if (object == nullptr) {
        PyErr_Print();
        exit(1);
      }
    }

    PyObject* PListFromVector(const vector<int>& nums, const string& type) {
      PyObject* l = PyList_New(nums.size());
      for (size_t i = 0; i < nums.size(); ++i) {
        if (type == "double") {
          PyList_SET_ITEM(l, i, PyFloat_FromDouble(nums[i]));
        } else {
          PyList_SET_ITEM(l, i, PyLong_FromLong(nums[i]));
        }
      }
      return l;
    }

    PyObject* PMatrixFromVector(const vector<vector<int>>& matrix, const string& type) {
      PyObject* l = PyList_New(matrix.size());
      for (size_t i = 0; i < matrix.size(); ++i) {
        PyList_SET_ITEM(l, i, PListFromVector(matrix[i], type));
      }
      return l;
    }

    int num_of_variables_;
    vector<int> bounds_;
    vector<vector<int>> ineqs_;
    vector<double> last_sol_;
};

PyObject* LPSolver::feasibility_func_ = nullptr;
PyObject* LPSolver::sol_func_ = nullptr;


void PrintVec(const vector<int>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    cout << v[i] << " ";
  }
  cout << endl;
}

class NashDigraph {
  public:
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
        if (is_complete) { // otherwise we will add (0, 0, ..., 0) costs
          for (int cost_idx = 0; cost_idx < num_of_players; ++cost_idx) {
            in >> edge_cost[cost_idx];
          }
        }
        AddEdge(v, u, edge_cost, edge_idx);
      }
      num_of_players_ = num_of_players;
      num_of_edges_ = num_of_edges;
      Preprocess();
      is_limited_by_iters_ = false;
    }

    // For this constructor edges should be added manually
    NashDigraph(const vector<int>& turns, int num_of_players, size_t start_vertex) : 
      turns_(turns),
      edges_(vector<vector<Edge>>(turns.size())),
      start_vertex_(start_vertex),
      num_of_players_(num_of_players),
      num_of_edges_(0) {
        is_limited_by_iters_ = false;
    }

    void Preprocess() {
      CalcAllPossiblePlayersStrategies();
    }

    LPSolver ConfigureBaseLP() {
      LPSolver res(num_of_edges_);
      for (size_t var_idx = 0; var_idx < num_of_edges_; ++var_idx) {
        vector<int> ineq(num_of_edges_);
        ineq[var_idx] = -1;
        res.PushInequality(ineq, -1);
      }
      for (size_t v = 1; v <= 6; ++v) {
        for (auto& edge : edges_[v]) {
          if (edge.finish != 0) {
            vector<int> ineq(num_of_edges_);
            size_t var_idx = edge.idx;
            ineq[var_idx] = 1;
            res.PushInequality(ineq, 1);
          }
        }
      }
      /*
      for (size_t var_idx = 0; var_idx < num_of_edges_; ++var_idx) {
        vector<int> ineq(num_of_edges_);
        ineq[var_idx] = 1;
        res.PushInequality(ineq, kEdgeCostLimit);
      }
      */
      return res;
    }

    void CalcImprovementsTable() {
      int profit = 0;

      size_t n = all_possible_players_strategies_[0].size();
      size_t m = all_possible_players_strategies_[1].size();
      num_of_fails_by_cell_ = vector<vector<int>>(n, vector<int>(m, 0));
      vector<vector<int>> can_improve_row(n, vector<int>(m, 0));
      vector<vector<int>> can_improve_col(n, vector<int>(m, 0));
      vector<vector<vector<int>>> linear_funcs_by_cell(n, vector<vector<int>>(m));
      for (int cx = 0; cx < n; ++cx) {
        for (int cy = 0; cy < m; ++cy) {
          vector<size_t> all_players_strategy(turns_.size());
          ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][cx], 0, &all_players_strategy);
          ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][cy], 1, &all_players_strategy);
          linear_funcs_by_cell[cx][cy] = GetLinFuncsForPlayersByGlobalStrategy(all_players_strategy);
        }
      }
    
      LPSolver base_lp = ConfigureBaseLP();
      for (size_t cx = 0; cx < n; ++cx) {
        for (size_t cy = 0; cy < m; ++cy) {
          if (linear_funcs_by_cell[cx][cy].empty()) {
            continue;
          }
          LPSolver tmp_lp = base_lp;
          for (size_t t = 0; t < m; ++t) {
            if (linear_funcs_by_cell[cx][t] == linear_funcs_by_cell[cx][cy]) {
              continue;
            }
            assert(AddInequality(linear_funcs_by_cell[cx][t], linear_funcs_by_cell[cx][cy], &tmp_lp));
          }

          can_improve_row[cx][cy] = tmp_lp.IsFeasible();
          profit += !can_improve_row[cx][cy];

          tmp_lp = base_lp;

          for (size_t t = 0; t < n; ++t) {
            if (linear_funcs_by_cell[t][cy] == linear_funcs_by_cell[cx][cy]) {
              continue;
            }
            assert(AddInequality(linear_funcs_by_cell[t][cy], linear_funcs_by_cell[cx][cy], &tmp_lp));
          }

          can_improve_col[cx][cy] = tmp_lp.IsFeasible();
          profit += !can_improve_col[cx][cy];
        }
      }

      row_jumps_ = vector<vector<int>>(n);
      for (int cx = 0; cx < n; ++cx) {
        map<vector<int>, int> occurs;
        for (int cy = 0; cy < m; ++cy) {
          occurs[linear_funcs_by_cell[cx][cy]] = cy;
        }
        for (auto it = occurs.begin(); it != occurs.end(); ++it) {
          int cy = it->second;
          if (can_improve_row[cx][cy]) {
            row_jumps_[cx].emplace_back(cy);
          }
        }
      }
      
      col_jumps_ = vector<vector<int>>(m);
      for (int cy = 0; cy < m; ++cy) {
        map<vector<int>, int> occurs;

        for (int cx = 0; cx < n; ++cx) {
          occurs[linear_funcs_by_cell[cx][cy]] = cx;
        }
        for (auto it = occurs.begin(); it != occurs.end(); ++it) {
          int cx = it->second;
          if (can_improve_col[cx][cy]) {
            col_jumps_[cy].emplace_back(cx);
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

      // all_possible_players_strategies_[player_idx].pop_back();
      // all_possible_players_strategies_[player_idx].pop_back();
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

    void CheckCorrectness() {
      vector<pair<int, int>> sol = GetSolution();
      for (size_t v = 0; v < turns_.size(); ++v) {
        for (auto& edge : edges_[v]) {
          assert(edge.idx < sol.size());
          cout << sol[edge.idx].first << " " << sol[edge.idx].second << endl;
          edge.cost = vector<int>({sol[edge.idx].first, sol[edge.idx].second});
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
            // assert(!is_pos_in_ne);
          }
          num_of_ne += is_pos_in_ne;
        }
      }
      int num_of_saturated_cells = ineq_sat_percentage_ *  n * m;
      cout << n * m - num_of_ne << " " << num_of_saturated_cells << endl;
      is_complete_ = false;
    }

    int CountNumOfNE() {
      size_t n = turns_.size();
      vector<int> num_of_strategies_limits(num_of_players_);
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        num_of_strategies_limits[player_idx] = all_possible_players_strategies_[player_idx].size();
      }
      vector<vector<int>> all_possible_strategies_corteges = GenAllPossibleChoices(num_of_strategies_limits);
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

    vector<int> GetLinFuncsForPlayersByGlobalStrategy(const vector<size_t>& all_players_strategy) {
      size_t curv = start_vertex_;
      vector<int> is_vertex_used(turns_.size());
      vector<int> used_edges;
      while (!is_vertex_used[curv] && turns_[curv] != -1) {
        is_vertex_used[curv] = 1;
        assert(curv < all_players_strategy.size());
        size_t index_of_edge_to_use = all_players_strategy[curv];
        assert(index_of_edge_to_use < edges_[curv].size());
        size_t nextv = edges_[curv][index_of_edge_to_use].finish;
        used_edges.emplace_back(edges_[curv][index_of_edge_to_use].idx);
        curv = nextv;
      }
      if (is_vertex_used[curv]) { // got cycle
        used_edges.clear();
        return used_edges;
      }
      vector<int> res(num_of_edges_);
      for (size_t edge_idx : used_edges) {
        assert(edge_idx < num_of_edges_);
        res[edge_idx] = 1;
      }
      return res;
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
      return lp_solver->PushInequality(ineq, -1);
    }

    bool GoFirstPlayer(
      const vector<vector<vector<int>>>& linear_funcs_by_cell,
      int cx, 
      int cy,
      vector<vector<int>>* is_cell_used, 
      LPSolver* lp_x,
      LPSolver* lp_y
    ) {
      int n = is_cell_used->size();
      // finding cell to improve for the first player
      for (int tx : col_jumps_[cy]) {
        size_t old_lp_solver_size = lp_x->Size();
        if (linear_funcs_by_cell[tx][cy] == linear_funcs_by_cell[cx][cy]) {
          continue;
        }
        const vector<int>& best_linear_func = linear_funcs_by_cell[tx][cy];
        assert(!best_linear_func.empty());
        for (int func_idx : col_jumps_[cy]) {
          if (func_idx == tx) {
            continue;
          }
          assert(AddInequality(linear_funcs_by_cell[func_idx][cy], linear_funcs_by_cell[tx][cy], lp_x));
        }
        if (lp_x->IsFeasible()) {
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
          bool branch_result = SolveTwoPlayersCostsRec(linear_funcs_by_cell, tx, cy, 1, is_cell_used, lp_x, lp_y);
          if (branch_result) {
            return true;
          }
          for (int colored_cell : colored_cells) {
            (*is_cell_used)[colored_cell][cy] = 0;
          }
        }
        while (lp_x->Size() != old_lp_solver_size) {
          lp_x->PopInequality();
        } 
      }
      return false;
    }

    bool GoSecondPlayer(
      const vector<vector<vector<int>>>& linear_funcs_by_cell,
      int cx, 
      int cy,
      vector<vector<int>>* is_cell_used, 
      LPSolver* lp_x,
      LPSolver* lp_y
    ) {
      int m = (*is_cell_used)[0].size();
      // finding cell to improve for the second player
      for (int ty : row_jumps_[cx]) {
        size_t old_lp_solver_size = lp_y->Size();
        if (linear_funcs_by_cell[cx][ty] == linear_funcs_by_cell[cx][cy]) { 
          continue;
        }
        const vector<int>& best_linear_func = linear_funcs_by_cell[cx][ty];
        assert(!best_linear_func.empty());
        for (int func_idx : row_jumps_[cx]) {
          if (func_idx == ty) {
            continue;
          }
          assert(AddInequality(linear_funcs_by_cell[cx][func_idx], linear_funcs_by_cell[cx][ty], lp_y));
        }
        if (lp_y->IsFeasible()) {
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
          bool branch_result = SolveTwoPlayersCostsRec(linear_funcs_by_cell, cx, ty, 0, is_cell_used, lp_x, lp_y);
          if (branch_result) {
            return true;
          }
          for (int colored_cell : colored_cells) {
            (*is_cell_used)[cx][colored_cell] = 0;
          }
        }
        while (lp_y->Size() != old_lp_solver_size) {
          lp_y->PopInequality();
        } 
      }
      return false;
    }

    bool SolveTwoPlayersCostsRec(
      const vector<vector<vector<int>>>& linear_funcs_by_cell,
      int cx, 
      int cy,
      int direction, // -1 for first cell, 0 - GoFirstPlayer, 1 - GoSecondPlayer
      vector<vector<int>>* is_cell_used, 
      LPSolver* lp_x,
      LPSolver* lp_y
    ) {
      if (is_limited_by_iters_) {
        if (num_of_transmissions_limit_ <= 0) {
          return false;
        }
        num_of_transmissions_limit_--;
      }
      int n = is_cell_used->size();
      int m = (*is_cell_used)[0].size();
      assert(cx < n);
      assert(cy < m);
      size_t num_of_used_cells = 0;
      for (size_t wx = 0; wx < n; ++wx) {
        for (size_t wy = 0; wy < m; ++wy) {
          if ((*is_cell_used)[wx][wy]) {
            num_of_used_cells++;
          }
        }
      }
      double sat_percentage_ = double(num_of_used_cells) / (n * m);
      cout << "Branch percentage: " << sat_percentage_ << endl;
      if (sat_percentage_ > ineq_sat_percentage_) {
        ineq_sat_percentage_ = sat_percentage_;
        best_solver_first_player_ = *lp_x;
        best_solver_second_player_ = *lp_y;
        best_cells_cover_matrix_ = *is_cell_used;
      }
      if ((*is_cell_used)[cx][cy]) {
        int wx, wy, max_num_of_fails = -1;
        for (int tx = 0; tx < n; ++tx) {
          for (int ty = 0; ty < m; ++ty) {
            if (!(*is_cell_used)[tx][ty]) {
              if (num_of_fails_by_cell_[tx][ty] > max_num_of_fails) {
                max_num_of_fails = num_of_fails_by_cell_[tx][ty];
                wx = tx;
                wy = ty;
              }
            }
          }
        }
        if (max_num_of_fails == -1) {
          return true;
        }
        return SolveTwoPlayersCostsRec(linear_funcs_by_cell, wx, wy, -1, is_cell_used, lp_x, lp_y);
      }
      (*is_cell_used)[cx][cy] = 1;
      // randomizing branch's order
      
      if (direction == -1) {
        bool res = GoFirstPlayer(linear_funcs_by_cell, cx, cy, is_cell_used, lp_x, lp_y);
        if (res) {
          return true;
        }
        res = GoSecondPlayer(linear_funcs_by_cell, cx, cy, is_cell_used, lp_x, lp_y);
        if (res) {
          return true;
        }
        num_of_fails_by_cell_[cx][cy]++;
        (*is_cell_used)[cx][cy] = 0;
        return false;
      }

      if (direction == 0) {
        bool res = GoFirstPlayer(linear_funcs_by_cell, cx, cy, is_cell_used, lp_x, lp_y);
        if (res) {
          return true;
        }
        num_of_fails_by_cell_[cx][cy]++;
        (*is_cell_used)[cx][cy] = 0;
        return false;
      }

      bool res = GoSecondPlayer(linear_funcs_by_cell, cx, cy, is_cell_used, lp_x, lp_y);
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

    bool SolveTwoPlayersCosts(bool are_costs_positive) {
      assert(num_of_players_ == 2);
      int n = all_possible_players_strategies_[0].size();
      int m = all_possible_players_strategies_[1].size();
      ineq_sat_percentage_ = -1.0;
      cerr << "Num of strategies for players: " << n << " " << m << endl;
      if (n == 0 || m == 0) {
        return false;
      }
      vector<vector<int>> is_pair_of_strategies_used(n, vector<int>(m));
      vector<vector<vector<int>>> linear_funcs_by_cell(
        n, 
        vector<vector<int>>(m) // if vector is empty, then both vectors should be empty and this is a cycle
      );
      for (int cx = 0; cx < n; ++cx) {
        for (int cy = 0; cy < m; ++cy) {
          vector<size_t> all_players_strategy(turns_.size());
          ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][cx], 0, &all_players_strategy);
          ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][cy], 1, &all_players_strategy);
          linear_funcs_by_cell[cx][cy] = GetLinFuncsForPlayersByGlobalStrategy(all_players_strategy);
        }
      }
      LPSolver lp_x;
      LPSolver lp_y;
      if (are_costs_positive) {
        lp_x = ConfigureBaseLP();
        lp_y = ConfigureBaseLP();
      }
      return SolveTwoPlayersCostsRec(linear_funcs_by_cell, 0, 0, -1, &is_pair_of_strategies_used, &lp_x, &lp_y);
    }

    bool SolveThreePlayersCosts() {
      int n = all_possible_players_strategies_[0].size();
      int m = all_possible_players_strategies_[1].size();
      int k = all_possible_players_strategies_[2].size();
      ineq_sat_percentage_ = -1.0;
      cerr << "Num of strategies for players: " << n << " " << m << " " << k << endl;
      if (n == 0 || m == 0 || k == 0) {
        return false;
      }
      vector<vector<vector<int>>> is_cell_used(n, vector<vector<int>>(m, vector<int>(k)));
      vector<vector<vector<vector<int>>>> linear_func_by_cell(n, vector<vector<vector<int>>>(m, vector<vector<int>>(k)));
      for (int cx = 0; cx < n; ++cx) {
        for (int cy = 0; cy < m; ++cy) {
          for (int cz = 0; cz < k; ++cz) {
            vector<size_t> all_players_strategy(turns_.size());
            ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[0][cx], 0, &all_players_strategy);
            ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[1][cy], 1, &all_players_strategy);
            ApplyPlayerStrategyToGlobalOne(all_possible_players_strategies_[2][cz], 2, &all_players_strategy);
            linear_func_by_cell[cx][cy][cz] = GetLinFuncsForPlayersByGlobalStrategy(all_players_strategy);
          }
        }
      }

      LPSolver lp_x(num_of_edges_), lp_y(num_of_edges_), lp_z(num_of_edges_);
      for (size_t var_idx = 0; var_idx < num_of_edges_; ++var_idx) {
        vector<int> ineq(num_of_edges_);
        ineq[var_idx] = -1;
        lp_x.PushInequality(ineq, -1);
        lp_y.PushInequality(ineq, -1);
        lp_z.PushInequality(ineq, -1);
      }      
      return SolveThreePlayersCostsRec(linear_func_by_cell, 0, 0, 0, &is_cell_used, &lp_x, &lp_y, &lp_z);
    }


    bool GoX(
      const vector<vector<vector<vector<int>>>>& linear_funcs_by_cell,
      int cx,
      int cy,
      int cz,
      vector<vector<vector<int>>>* is_cell_used,
      LPSolver* lp_x,
      LPSolver* lp_y,
      LPSolver* lp_z
    ) {
      rec_stack.push_back('X');
      int n = is_cell_used->size();
      for (int tx = 0; tx < n; ++tx) {
        size_t old_lp_solver_size = lp_x->Size();
        if (linear_funcs_by_cell[tx][cy][cz] == linear_funcs_by_cell[cx][cy][cz]) { 
          continue;
        }
        
        const vector<int>& best_linear_func = linear_funcs_by_cell[tx][cy][cz];
        if (best_linear_func.empty()) {
          continue;
        }
        for (int func_idx = 0; func_idx < n; ++func_idx) {
          if (linear_funcs_by_cell[func_idx][cy][cz] == linear_funcs_by_cell[tx][cy][cz]) {
            continue;
          }
          assert(AddInequality(linear_funcs_by_cell[func_idx][cy][cz], linear_funcs_by_cell[tx][cy][cz], lp_x));
        }
        if (lp_x->IsFeasible()) {
          vector<int> colored_cells;
          for (int wx = 0; wx < n; ++wx) {
            if (linear_funcs_by_cell[tx][cy][cz] == linear_funcs_by_cell[wx][cy][cz]) {
              continue;
            }
            if (!(*is_cell_used)[wx][cy][cz]) {
              colored_cells.emplace_back(wx);
              (*is_cell_used)[wx][cy][cz] = 1;
            }
          }
          bool branch_result = SolveThreePlayersCostsRec(linear_funcs_by_cell, tx, cy, cz, is_cell_used, lp_x, lp_y, lp_z);
          if (branch_result) {
            return true;
          }
          for (int colored_cell : colored_cells) {
            (*is_cell_used)[colored_cell][cy][cz] = 0;
          }
        }
      
        while (lp_x->Size() != old_lp_solver_size) {
          lp_x->PopInequality();
        } 
      }
      rec_stack.pop_back();
      return false;
    }

    bool GoY(
      const vector<vector<vector<vector<int>>>>& linear_funcs_by_cell,
      int cx,
      int cy,
      int cz,
      vector<vector<vector<int>>>* is_cell_used,
      LPSolver* lp_x,
      LPSolver* lp_y,
      LPSolver* lp_z
    ) {
      rec_stack.push_back('Y');
      int m = (*is_cell_used)[0].size();
      for (int ty = 0; ty < m; ++ty) {
        size_t old_lp_solver_size = lp_y->Size();
        if (linear_funcs_by_cell[cx][ty][cz] == linear_funcs_by_cell[cx][cy][cz]) { 
          continue;
        }
        
        const vector<int>& best_linear_func = linear_funcs_by_cell[cx][ty][cz];
        if (best_linear_func.empty()) {
          continue;
        }
        for (int func_idx = 0; func_idx < m; ++func_idx) {
          if (linear_funcs_by_cell[cx][func_idx][cz] == linear_funcs_by_cell[cx][ty][cz]) {
            continue;
          }
          assert(AddInequality(linear_funcs_by_cell[cx][func_idx][cz], linear_funcs_by_cell[cx][ty][cz], lp_y));
        }
        if (lp_y->IsFeasible()) {
          vector<int> colored_cells;
          for (int wy = 0; wy < m; ++wy) {
            if (linear_funcs_by_cell[cx][wy][cz] == linear_funcs_by_cell[cx][ty][cz]) {
              continue;
            }
            if (!(*is_cell_used)[cx][wy][cz]) {
              colored_cells.emplace_back(wy);
              (*is_cell_used)[cx][wy][cz] = 1;
            }
          }
          bool branch_result = SolveThreePlayersCostsRec(linear_funcs_by_cell, cx, ty, cz, is_cell_used, lp_x, lp_y, lp_z);
          if (branch_result) {
            return true;
          }
          for (int colored_cell : colored_cells) {
            (*is_cell_used)[cx][colored_cell][cz] = 0;
          }
        }
      
        while (lp_y->Size() != old_lp_solver_size) {
          lp_y->PopInequality();
        } 
      }
      rec_stack.pop_back();
      return false;
    }

    bool GoZ(
      const vector<vector<vector<vector<int>>>>& linear_funcs_by_cell,
      int cx,
      int cy,
      int cz,
      vector<vector<vector<int>>>* is_cell_used,
      LPSolver* lp_x,
      LPSolver* lp_y,
      LPSolver* lp_z
    ) {
      rec_stack.push_back('Z');
      int k = (*is_cell_used)[0][0].size();
      for (int tz = 0; tz < k; ++tz) {
        size_t old_lp_solver_size = lp_z->Size();
        if (linear_funcs_by_cell[cx][cy][tz] == linear_funcs_by_cell[cx][cy][cz]) { 
          continue;
        }
        
        const vector<int>& best_linear_func = linear_funcs_by_cell[cx][cy][tz];
        if (best_linear_func.empty()) {
          continue;
        }
        for (int func_idx = 0; func_idx < k; ++func_idx) {
          if (linear_funcs_by_cell[cx][cy][func_idx] == linear_funcs_by_cell[cx][cy][tz]) {
            continue;
          }
          assert(AddInequality(linear_funcs_by_cell[cx][cy][func_idx], linear_funcs_by_cell[cx][cy][tz], lp_z));
        }
        if (lp_z->IsFeasible()) {
          vector<int> colored_cells;
          for (int wz = 0; wz < k; ++wz) {
            if (linear_funcs_by_cell[cx][cy][wz] == linear_funcs_by_cell[cx][cy][tz]) {
              continue;
            }
            if (!(*is_cell_used)[cx][cy][wz]) {
              colored_cells.emplace_back(wz);
              (*is_cell_used)[cx][cy][wz] = 1;
            }
          }
          bool branch_result = SolveThreePlayersCostsRec(linear_funcs_by_cell, cx, cy, tz, is_cell_used, lp_x, lp_y, lp_z);
          if (branch_result) {
            return true;
          }
          for (int colored_cell : colored_cells) {
            (*is_cell_used)[cx][cy][colored_cell] = 0;
          }
        }
      
        while (lp_z->Size() != old_lp_solver_size) {
          lp_z->PopInequality();
        } 
      }
      rec_stack.pop_back();
      return false;
    }

    bool SolveThreePlayersCostsRec(
      const vector<vector<vector<vector<int>>>>& linear_funcs_by_cell,
      int cx,
      int cy,
      int cz,
      vector<vector<vector<int>>>* is_cell_used,
      LPSolver* lp_x,
      LPSolver* lp_y,
      LPSolver* lp_z
    ) {
      int n = is_cell_used->size();
      int m = (*is_cell_used)[0].size();
      int k = (*is_cell_used)[0][0].size();

      if ((*is_cell_used)[cx][cy][cz]) {
        for (int tx = 0; tx < n; ++tx) {
          for (int ty = 0; ty < m; ++ty) {
            for (int tz = 0; tz < k; ++tz) {
              if (!(*is_cell_used)[tx][ty][tz]) {
                return SolveThreePlayersCostsRec(linear_funcs_by_cell, tx, ty, tz, is_cell_used, lp_x, lp_y, lp_z);
              }
            }
          }
        }
        return true;
      }
      size_t num_of_used_cells = 0;
      for (int tx = 0; tx < n; ++tx) {
        for (int ty = 0; ty < m; ++ty) {
          for (int tz = 0; tz < k; ++tz) {
            if ((*is_cell_used)[tx][ty][tz]) {
              num_of_used_cells++;
            }
          }
        }
      }
      cout << double(num_of_used_cells) / (n * m * k) << endl;
      // cout << rec_stack << endl;
      (*is_cell_used)[cx][cy][cz] = 1;

      vector<function<bool()>> branch_calls;

      branch_calls.emplace_back([this, &linear_funcs_by_cell, &cx, &cy, &cz, &is_cell_used, &lp_x, &lp_y, &lp_z]() { 
        return GoY(linear_funcs_by_cell, cx, cy, cz, is_cell_used, lp_x, lp_y, lp_z); 
      });

      branch_calls.emplace_back([this, &linear_funcs_by_cell, &cx, &cy, &cz, &is_cell_used, &lp_x, &lp_y, &lp_z]() { 
        return GoX(linear_funcs_by_cell, cx, cy, cz, is_cell_used, lp_x, lp_y, lp_z); 
      });

      branch_calls.emplace_back([this, &linear_funcs_by_cell, &cx, &cy, &cz, &is_cell_used, &lp_x, &lp_y, &lp_z]() { 
        return GoZ(linear_funcs_by_cell, cx, cy, cz, is_cell_used, lp_x, lp_y, lp_z); 
      });
        
      Shuffle(&branch_calls);

      for (auto& branch_call : branch_calls) {
        bool res = branch_call();
        if (res) {
          return true;
        }
      }

      (*is_cell_used)[cx][cy][cz] = 0;
      return false;
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

    void SetTransmissionsLimit(int x) {
      is_limited_by_iters_ = true;
      num_of_transmissions_limit_ = x;
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

    string rec_stack;
    vector<vector<int>> num_of_fails_by_cell_;
    vector<vector<int>> row_jumps_;
    vector<vector<int>> col_jumps_;
    int num_of_transmissions_limit_;
    vector<int> turns_; // each value is in [-1; num_of_players), where -1 denotes terminal vertex
    vector<vector<Edge>> edges_;
    size_t start_vertex_;
    size_t num_of_players_;
    size_t num_of_edges_;
    vector<vector<vector<int>>> all_possible_players_strategies_;
    bool is_complete_; // are costs of edges added?
    bool is_limited_by_iters_;
    double ineq_sat_percentage_;
    LPSolver best_solver_first_player_; // used only for SolveTwoPlayersRec 
    LPSolver best_solver_second_player_; // used only for SolveTwoPlayersRec
    vector<vector<int>> best_cells_cover_matrix_; // used only for SolveTwoPlayersRec
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

void GenAllPossibleChoicesForMasksRec(const vector<int>& limits, const vector<pair<int, int>>& bounds, vector<int>* cur_cortege, vector<vector<int>>* res) {
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

bool TryToSolve(int ps_lb, int ps_rb, int cycle_size, const std::vector<pair<int, int>>& bounds, const std::string& offset_filename, bool should_shuffle) {
  ifstream in(offset_filename);
  assert(in.is_open());
  int offset = 0;
  in >> offset;
  in.close();

  const size_t kDumpProgressPeriod = 10;
  
  double max_ineq_sat_percentage = 0.0;
  size_t best_graph_id = 0;
  vector<GraphId> graph_ids_to_check;
  vector<vector<int>> choices_to_connect_with_cycle;
  vector<vector<int>> choices_to_build_path;
  for (int path_size = ps_lb; path_size <= ps_rb; ++path_size) {

    vector<int> path_to_cycle_edges_ways_limits(path_size, (1 << cycle_size));
    choices_to_connect_with_cycle = GenAllPossibleChoicesForMasks(path_to_cycle_edges_ways_limits, bounds);
   
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
      if (vertex_in_path != 0) {
        AddEdge(cycle_size + 1 + vertex_in_path, 0, &edges); // edge to terminal from prefix vertex except start
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
    if (!G.AreAllVerticesAccessibleFromStart() || !G.HasOnlyOneTerminal()) {
      cerr << "This graph will be skipped, as not all vertices are accessible from start" << endl;
      continue;
    }
    G.Print(false);
    G.Preprocess();
    G.CalcImprovementsTable();
    // G.SetTransmissionsLimit(1000);
    bool g_res = G.SolveTwoPlayersCosts(true);
    G.CheckCorrectness();
    double cur_ineq_sat_percentage = G.GetIneqSatPercentage();
    if (cur_ineq_sat_percentage > max_ineq_sat_percentage) {
      best_graph_id = graph_id_idx;
      max_ineq_sat_percentage = cur_ineq_sat_percentage;
    }
    max_ineq_sat_percentage = max(max_ineq_sat_percentage, G.GetIneqSatPercentage());
    cout << "Current max inequality saturation percentage: " << max_ineq_sat_percentage << endl;
    if (g_res) {
      return true;
    }
    if (graph_id_idx % kDumpProgressPeriod == 0) {
      ofstream out(offset_filename);
      assert(out.is_open());
      out << graph_id_idx << "\n";
      out << max_ineq_sat_percentage << "\n";
      out << best_graph_id << "\n";
      out.close();
    }
  }
  return false;
}

int main() {
  LPSolver::LaunchPython();
  //freopen("input.txt", "r", stdin);
  //NashDigraph G("input.txt", false);
  //cout << G.SolveThreePlayersCosts() << endl;
  //G.CheckCorrectnessThree();

  // cout << G.AreAllVerticesAccessibleFromStart() << endl;
  // cout << G.SolveTwoPlayersCosts(true) << endl;
  // G.CheckCorrectness();
  //cout << G.GetIneqSatPercentage() << endl;
  //cout << G.CountNumOfNE() << endl;
  TryToSolve(2, 3, 6, {{0, 2}, {0, 2}, {0, 2}, {0, 0}}, "offset.txt", true);
  //cout << TryToSolve(2, 3, 4, {{0, 2}, {0, 2}, {0, 0}, {0, 2}, {0, 0}}, "offset.txt", true) << endl; //offset - 1732 // 0.991 930
  // 2250 - for cycle_size = 3
  // 320 for {3, 3, 3} and cycle_size = 6
  //cout << TryToSolve(2, 3, 6, 3, 208, true);

  LPSolver::ReleasePython();
  return 0;
}