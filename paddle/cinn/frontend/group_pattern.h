#pragma once

#include <unordered_map>
#include <atomic>
#include <vector>
#include "paddle/cinn/api/op_topo_pattern.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::frontend {

struct FrontendPattern {};

}

namespace cinn::api{
  struct ShardableAxis {
    int axis;
    std::string axis_name;

    static int64_t UnqiueSeqNo() {
      static std::atomic<int64_t> cnt(0);
      return ++cnt;
    }
  };

  using ShardableAxes = std::vector<ShardableAxis>;

  struct ShardableAxesSignature {
    using OpOperand = std::pair<const pir::Operation*, /*operand index*/int>;

    ShardableAxes output_shardable_axes;
    std::unordered_map<OpOperand, ShardableAxes> input_shardable_axes;
  };

}

namespace cinn::api {

template<>
struct ErrorPattern<frontend::FrontendPattern> {
  explicit ErrorPattern(const ErrorPattern<frontend::FrontendPatterns>& other) = default;

  const pir::Operation* op;
  std::string error_string;
};

template<>
struct InjectiveSourcePattern<frontend::FrontendPattern> {
  explicit InjectiveSourcePattern(const InjectiveSourcePattern<frontend::FrontendPatterns>& other) = default;
  std::vector<const pir::Operation*> ops;
};

template<>
struct SingleReductionOpPattern<frontend::FrontendPattern> {
  explicit SingleReductionOpPattern(const SingleReductionOpPattern<frontend::FrontendPatterns>& other) = default;
  const pir::Operation* reduce_op;
};

template<>
struct ReductionPattern<frontend::FrontendPattern> {
  explicit ReductionPattern(const ReductionPattern<frontend::FrontendPatterns>& other) = default;
};

template<>
struct PartialShardablePattern<frontend::FrontendPattern> {
  explicit PartialShardablePattern(const PartialShardablePattern<frontend::FrontendPatterns>& other) = default;
  std::vector<const pir::Operation*> ops;
  ShardableAxesSignature shardable_axes_signature;
};

}

namespace cinn::frontend {

using StmtPattern = api::StmtPattern<FrontendPattern>;
using ErrorGroupPattern = api::ErrorPattern<FrontendPattern>;

using GroupPattern = api::OpTopoPattern<FrontendPattern>;


}