"""
Optuna 演算法搜尋實作
使用 TPE / NSGA-II / Random 搜尋最佳量化配置

搜尋空間慣例（對應 search_space.py）：
  tuple (low, high)   → suggest_float（連續，TPE 可充分探索）
  list  [...]         → suggest_categorical（只有這幾個合法值）
"""

import logging
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from Global_Tuner_v2.modular_agent.schemas import StrategySuggestion
from .search_space import (
    ASVD_SPACE, GPTQ_SPACE, AWQ_SPACE, QQQ_SPACE, BNB_SPACE,
    SPARSE_UNSTRUCTURED_SPACE, SPARSE_STRUCTURED_SPACE, ALL_MODES,
)

logger = logging.getLogger("OptunaSearcher")


def _suggest(trial: optuna.Trial, name: str, space):
    """
    根據 space 型別自動選擇 suggest 方式：
      tuple (low, high)         → suggest_float，linear scale
      tuple (low, high, "log")  → suggest_float，log scale
      list  [...]               → suggest_categorical
    """
    if isinstance(space, tuple):
        if len(space) == 3 and space[2] == "log":
            return trial.suggest_float(name, space[0], space[1], log=True)
        return trial.suggest_float(name, space[0], space[1])
    return trial.suggest_categorical(name, space)


class OptunaSearcher:
    """
    以 Optuna 的 ask-and-tell 模式逐步建議量化配置。

    sampler:  "tpe" (預設) | "nsga2" | "random"
    modes:    要包含的模式清單（None = 全部）
    seed:     亂數種子（可重現性）
    """

    def __init__(
        self,
        sampler: str = "tpe",
        modes: Optional[List[str]] = None,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
        population_size: int = 50,
    ):
        self.modes = modes or ALL_MODES
        self.sampler = sampler

        if sampler == "tpe":
            _sampler = optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                n_startup_trials=n_startup_trials,
            )
        elif sampler == "nsga2":
            _sampler = optuna.samplers.NSGAIISampler(
                seed=seed,
                population_size=population_size,
            )
        elif sampler == "random":
            _sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"未知的 sampler: {sampler}，可選 tpe/nsga2/random")

        self.study = optuna.create_study(direction="maximize", sampler=_sampler)
        self._pending_trial: Optional[optuna.Trial] = None
        logger.info(f"Optuna study 建立完成 (sampler={sampler}, modes={self.modes})")

    # ──────────────────────────────────────────────────────────────────────────
    def _trial_to_suggestion(self, trial: optuna.Trial) -> StrategySuggestion:
        """將 Optuna trial 的建議值對應至 StrategySuggestion"""
        mode_key = trial.suggest_categorical("mode", self.modes)
        kwargs: dict = {"reasoning": f"optuna:{mode_key}", "mode": mode_key}

        # ── ASVD only ────────────────────────────────────────────────────────
        if mode_key == "asvd_only":
            kwargs["alpha"]              = _suggest(trial, "alpha",              ASVD_SPACE["alpha"])
            kwargs["param_ratio_target"] = _suggest(trial, "param_ratio_target", ASVD_SPACE["param_ratio_target"])
            kwargs["scaling_method"]     = _suggest(trial, "scaling_method",     ASVD_SPACE["scaling_method"])

        # ── GPTQ ─────────────────────────────────────────────────────────────
        elif mode_key == "gptq":
            kwargs["mode"]           = "quant_only"
            kwargs["quant_method"]   = "gptq"
            kwargs["quant_bits"]     = _suggest(trial, "gptq_bits",       GPTQ_SPACE["quant_bits"])
            kwargs["quant_group_size"] = _suggest(trial, "gptq_group_size", GPTQ_SPACE["quant_group_size"])
            kwargs["quant_format"]   = _suggest(trial, "gptq_format",     GPTQ_SPACE["quant_format"])
            kwargs["damp_percent"]   = _suggest(trial, "gptq_damp",       GPTQ_SPACE["damp_percent"])
            kwargs["mse"]            = 0.0

        # ── AWQ ──────────────────────────────────────────────────────────────
        elif mode_key == "awq":
            kwargs["mode"]             = "quant_only"
            kwargs["quant_method"]     = "awq"
            kwargs["quant_bits"]       = 4
            kwargs["quant_group_size"] = _suggest(trial, "awq_group_size", AWQ_SPACE["quant_group_size"])

        # ── QQQ ──────────────────────────────────────────────────────────────
        elif mode_key == "qqq":
            kwargs["mode"]             = "quant_only"
            kwargs["quant_method"]     = "qqq"
            kwargs["quant_bits"]       = 4
            kwargs["quant_format"]     = "qqq"
            kwargs["quant_group_size"] = _suggest(trial, "qqq_group_size", QQQ_SPACE["quant_group_size"])
            kwargs["damp_percent"]     = _suggest(trial, "qqq_damp",       QQQ_SPACE["damp_percent"])

        # ── BNB ──────────────────────────────────────────────────────────────
        elif mode_key == "bnb":
            kwargs["mode"]         = "quant_only"
            kwargs["quant_method"] = "bnb"
            bits = _suggest(trial, "bnb_bits", BNB_SPACE["quant_bits"])
            kwargs["quant_bits"]   = bits
            kwargs["use_double_quant"] = (
                _suggest(trial, "bnb_double_quant", BNB_SPACE["use_double_quant"])
                if bits == 4 else False
            )

        # ── Sparse Unstructured ───────────────────────────────────────────────
        elif mode_key == "sparse_unstructured":
            kwargs["mode"]              = "sparse_only"
            kwargs["sparsity_structure"] = "unstructured"
            kwargs["sparsity_ratio"]    = _suggest(trial, "sparse_ratio", SPARSE_UNSTRUCTURED_SPACE["sparsity_ratio"])

        # ── Sparse Structured ─────────────────────────────────────────────────
        elif mode_key == "sparse_structured":
            kwargs["mode"]               = "sparse_only"
            kwargs["sparsity_structure"] = _suggest(trial, "sparse_structure", SPARSE_STRUCTURED_SPACE["sparsity_structure"])

        # ── Hybrid: ASVD + BNB ───────────────────────────────────────────────
        elif mode_key == "hybrid_asvd_bnb":
            kwargs["mode"]               = "hybrid"
            kwargs["alpha"]              = _suggest(trial, "h_asvd_alpha",   ASVD_SPACE["alpha"])
            kwargs["param_ratio_target"] = _suggest(trial, "h_asvd_ratio",   ASVD_SPACE["param_ratio_target"])
            kwargs["scaling_method"]     = _suggest(trial, "h_asvd_scaling", ASVD_SPACE["scaling_method"])
            kwargs["quant_method"]       = "bnb"
            h_bnb_bits = _suggest(trial, "h_bnb_bits", BNB_SPACE["quant_bits"])
            kwargs["quant_bits"]         = h_bnb_bits
            kwargs["use_double_quant"]   = (
                _suggest(trial, "h_bnb_double_quant", BNB_SPACE["use_double_quant"])
                if h_bnb_bits == 4 else False
            )

        return StrategySuggestion(**kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    def get_suggestion(self, iteration: int, trial_history: list, **kwargs):
        """回傳 (StrategySuggestion, raw_dict)"""
        trial = self.study.ask()
        self._pending_trial = trial
        suggestion = self._trial_to_suggestion(trial)
        logger.info(f"Optuna 建議 [{iteration}]: mode={suggestion.mode} / "
                    f"quant={suggestion.quant_method}")
        return suggestion, suggestion.to_log_dict()

    def report_score(self, score: float):
        """回報本次 trial 的得分，讓 Optuna 更新模型"""
        if self._pending_trial is not None:
            self.study.tell(self._pending_trial, score)
            self._pending_trial = None

    def report_failure(self):
        """回報本次 trial 失敗"""
        if self._pending_trial is not None:
            self.study.tell(self._pending_trial, optuna.trial.TrialState.FAIL)
            self._pending_trial = None

    def add_past_trial(self, config: dict, score: float):
        """
        將已完成的 trial 重新注入 Optuna study，讓 TPE/NSGA-II 保有過去的先驗知識。
        在 resume 時呼叫，每個 history 中的 trial 都要 replay 一次。
        """
        from optuna.distributions import CategoricalDistribution, FloatDistribution
        from optuna.trial import create_trial

        # config 是巢狀結構：{"mode": ..., "asvd": {...}, "quant": {...}, "sparse": {...}}
        mode = config.get("mode", "")
        asvd  = config.get("asvd")  or {}
        quant = config.get("quant") or {}
        sparse = config.get("sparse") or {}

        # 反推 Optuna mode_key
        if mode == "asvd_only":
            mode_key = "asvd_only"
        elif mode == "quant_only":
            mode_key = quant.get("method", "")   # "gptq" / "awq" / "qqq" / "bnb"
        elif mode == "sparse_only":
            struct = sparse.get("structure", "")
            mode_key = "sparse_unstructured" if (not struct or struct == "unstructured") else "sparse_structured"
        elif mode == "hybrid":
            mode_key = "hybrid_asvd_bnb"
        else:
            return

        if mode_key not in self.modes:
            return

        params = {"mode": mode_key}
        dists  = {"mode": CategoricalDistribution(self.modes)}

        if mode_key == "asvd_only":
            if None in (asvd.get("alpha"), asvd.get("ratio"), asvd.get("scaling")):
                return
            params["alpha"]              = asvd["alpha"]
            dists["alpha"]               = CategoricalDistribution([0.3, 0.4, 0.5, 0.6, 0.7])
            params["param_ratio_target"] = asvd["ratio"]
            dists["param_ratio_target"]  = FloatDistribution(0.70, 0.99)
            params["scaling_method"]     = asvd["scaling"]
            dists["scaling_method"]      = CategoricalDistribution(["abs_mean", "abs_max", "fisher"])

        elif mode_key == "gptq":
            if None in (quant.get("bits"), quant.get("group_size"), quant.get("format"), quant.get("damp")):
                return
            params["gptq_bits"]       = quant["bits"]
            dists["gptq_bits"]        = CategoricalDistribution([3, 4, 8])
            params["gptq_group_size"] = quant["group_size"]
            dists["gptq_group_size"]  = CategoricalDistribution([16, 32, 64, 128, 256])
            params["gptq_format"]     = quant["format"]
            dists["gptq_format"]      = CategoricalDistribution(["gptq", "gptq_v2"])
            params["gptq_damp"]       = quant["damp"]
            dists["gptq_damp"]        = FloatDistribution(0.001, 0.1, log=True)

        elif mode_key == "awq":
            if quant.get("group_size") is None:
                return
            params["awq_group_size"] = quant["group_size"]
            dists["awq_group_size"]  = CategoricalDistribution([16, 32, 64, 128])

        elif mode_key == "qqq":
            if None in (quant.get("group_size"), quant.get("damp")):
                return
            params["qqq_group_size"] = quant["group_size"]
            dists["qqq_group_size"]  = CategoricalDistribution([-1, 128])
            params["qqq_damp"]       = quant["damp"]
            dists["qqq_damp"]        = FloatDistribution(0.0005, 0.05, log=True)

        elif mode_key == "bnb":
            if quant.get("bits") is None:
                return
            params["bnb_bits"]         = quant["bits"]
            dists["bnb_bits"]          = CategoricalDistribution([4, 8])
            params["bnb_double_quant"] = quant.get("double_quant", False)
            dists["bnb_double_quant"]  = CategoricalDistribution([False, True])

        elif mode_key == "sparse_unstructured":
            if sparse.get("ratio") is None:
                return
            params["sparse_ratio"] = sparse["ratio"]
            dists["sparse_ratio"]  = FloatDistribution(0.3, 0.7)

        elif mode_key == "sparse_structured":
            if sparse.get("structure") is None:
                return
            params["sparse_structure"] = sparse["structure"]
            dists["sparse_structure"]  = CategoricalDistribution(["2:4", "4:8"])

        elif mode_key == "hybrid_asvd_bnb":
            if None in (asvd.get("alpha"), asvd.get("ratio"), asvd.get("scaling"), quant.get("bits")):
                return
            params["h_asvd_alpha"]       = asvd["alpha"]
            dists["h_asvd_alpha"]        = CategoricalDistribution([0.3, 0.4, 0.5, 0.6, 0.7])
            params["h_asvd_ratio"]       = asvd["ratio"]
            dists["h_asvd_ratio"]        = FloatDistribution(0.70, 0.99)
            params["h_asvd_scaling"]     = asvd["scaling"]
            dists["h_asvd_scaling"]      = CategoricalDistribution(["abs_mean", "abs_max", "fisher"])
            params["h_bnb_bits"]         = quant["bits"]
            dists["h_bnb_bits"]          = CategoricalDistribution([4, 8])
            params["h_bnb_double_quant"] = quant.get("double_quant", False)
            dists["h_bnb_double_quant"]  = CategoricalDistribution([False, True])

        frozen = create_trial(params=params, distributions=dists, value=score)
        self.study.add_trial(frozen)

    def best_params(self) -> Optional[dict]:
        """回傳到目前為止最佳的參數（若有）"""
        try:
            return self.study.best_params
        except ValueError:
            return None
