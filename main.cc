#pragma comment(linker, "/STACK: 2000000")

#include <bits/stdc++.h>

using namespace std;

int PlusInf() {
  return numeric_limits<int>::max();
}

int MinusInf() {
  return numeric_limits<int>::lowest();
}

// inside [l; r]
int GetRandomInt(int l, int r) {
  assert(l <= r);
  static mt19937 mt;
  size_t next_token = mt();
  int len = r - l + 1;
  return l + next_token % len;
}

void GenAllPossibleChoices(const vector<int>& limits, vector<int>* cur_cortege, vector<vector<int>>* res) {
  size_t cur_idx = cur_cortege->size();
  if (cur_idx == limits.size()) {
    res->emplace_back(*cur_cortege);
    return;
  }
  for (size_t cur_elem = 0; cur_elem < limits[cur_idx]; cur_elem++) {
    cur_cortege->emplace_back(cur_elem);
    GenAllPossibleChoices(limits, cur_cortege, res);
    cur_cortege->pop_back();
  }
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
      cout << "Feasible check for " << inequalities_.size() <<  " inequalities" << endl;
      ofstream out("ineq.txt");
      assert(out.is_open());
      for (size_t ineq_idx = 0; ineq_idx < inequalities_.size(); ++ineq_idx) {
        for (size_t variable_idx = 0; variable_idx <= num_of_variables_; ++variable_idx) {
          out << inequalities_[ineq_idx][variable_idx] << " ";
        }
        out << "\n";
      }
      out.close();
      int ret_code = system("python lp_solver.py");
      assert(ret_code >= 0);
      ifstream res_file("res.txt");
      assert(res_file.is_open());
      int res_flag;
      res_file >> res_flag;
      if (res_flag == -1) {
        cout << "WARNING! Bad res flag" << endl;
      }
      return (res_flag == 1);
    }

  private:
    int num_of_variables_;
    vector<vector<int>> inequalities_; // the form of each inequality: a_1x_1 + a_2x_2 + ... + a_nx_n >= 0
};

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
      num_of_players_(num_of_players) {
    }

    void Preprocess() {
      CalcAllPossiblePlayersStrategies();
    }

    void AddEdge(int v, int u, const vector<int>& costs, int edge_idx) {
      edges_[v].push_back(Edge{u, costs, edge_idx});
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
      vector<int> tmp;
      GenAllPossibleChoices(player_num_of_edges_limits, &tmp, &all_possible_players_strategies_[player_idx]);
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
      vector<int> tmp;
      vector<int> num_of_strategies_limits(num_of_players_);
      for (size_t player_idx = 0; player_idx < num_of_players_; ++player_idx) {
        num_of_strategies_limits[player_idx] = all_possible_players_strategies_[player_idx].size();
      }
      vector<vector<int>> all_possible_strategies_corteges;
      GenAllPossibleChoices(num_of_strategies_limits, &tmp, &all_possible_strategies_corteges);
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
      LPSolver* lp_solver,
      bool is_max_improvement_unique
    ) {
      cout << "Branch" << endl;
      int n = is_cell_used->size();
      int m = (*is_cell_used)[0].size();
      assert(cx < n);
      assert(cy < m);
      if ((*is_cell_used)[cx][cy]) {
        for (int tx = 0; tx < n; ++tx) {
          for (int ty = 0; ty < m; ++ty) {
            if (!(*is_cell_used)[tx][ty]) {
              return SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, tx, ty, is_cell_used, lp_solver, is_max_improvement_unique);
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
        if (is_max_improvement_unique) {
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
            bool branch_result = SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, cx, cy, is_cell_used, lp_solver, is_max_improvement_unique);
            if (branch_result) {
              return true;
            }
            for (const auto& colored_cell : colored_cells) {
              (*is_cell_used)[colored_cell.first][colored_cell.second] = 0;
            }
          }
        } else {
          const vector<int>& cur_linear_func = linear_funcs_by_cell[cx][cy].first;
          const vector<int>& next_linear_func = linear_funcs_by_cell[tx][cy].first;
          bool can_be_improved = AddInequality(cur_linear_func, next_linear_func, lp_solver);
          if (can_be_improved && lp_solver->IsFeasible()) {
            bool branch_result = SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, tx, cy, is_cell_used, lp_solver, false);
            if (branch_result) {
              return true;
            }
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
        if (is_max_improvement_unique) {
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
            bool branch_result = SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, cx, cy, is_cell_used, lp_solver, is_max_improvement_unique);
            if (branch_result) {
              return true;
            }
            for (const auto& colored_cell : colored_cells) {
              (*is_cell_used)[colored_cell.first][colored_cell.second] = 0;
            }
          }
        } else {
          const vector<int>& cur_linear_func = linear_funcs_by_cell[cx][cy].first;
          const vector<int>& next_linear_func = linear_funcs_by_cell[cx][ty].first;
          bool can_be_improved = AddInequality(cur_linear_func, next_linear_func, lp_solver);
          if (can_be_improved && lp_solver->IsFeasible()) {
            bool branch_result = SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, cx, ty, is_cell_used, lp_solver, false);
            if (branch_result) {
              return true;
            }
          }
        }
        while (lp_solver->Size() != old_lp_solver_size) {
          lp_solver->PopInequality();
        } 
      }
      (*is_cell_used)[cx][cy] = 0;
      return false;
    }

    bool SolveTwoPlayersPositiveCosts(bool is_max_improvement_unique) {
      assert(num_of_players_ == 2);
      int n = all_possible_players_strategies_[0].size();
      int m = all_possible_players_strategies_[1].size();
      cerr << n << " " << m << endl;
      char ch;
      cin >> ch;
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
      return SolveTwoPlayersPositiveCostsRec(linear_funcs_by_cell, 0, 0, &is_pair_of_strategies_used, &lp_solver, is_max_improvement_unique);
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
};

int main() {
    //freopen("input.txt", "r", stdin);
    NashDigraph G("input.txt", false);
    cout << G.SolveTwoPlayersPositiveCosts(true) << endl;
    //cout << G.CountNumOfNE() << endl;

    return 0;
}