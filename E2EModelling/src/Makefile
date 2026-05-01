.PHONY: all features baseline forecast hypertune lstm clean-logs help

PYTHON ?= python
LOG_DIR := logs

define RUN_STAGE
	powershell -NoProfile -ExecutionPolicy Bypass -Command "New-Item -ItemType Directory -Force '$(LOG_DIR)' | Out-Null; $$ts=Get-Date -Format 'yyyyMMdd_HHmmss'; $$log='$(LOG_DIR)/_$(1)_' + $$ts + '.log'; & '$(PYTHON)' '$(2)' *> $$log; $$code=$$LASTEXITCODE; Write-Host ('Log: ' + $$log); exit $$code"
endef

all: features baseline forecast

features:
	$(call RUN_STAGE,features,feature_engineering/build_features.py)

baseline:
	$(call RUN_STAGE,baseline,modelling/baseline.py)

forecast:
	$(call RUN_STAGE,forecast,modelling/forecast.py)

hypertune:
	$(call RUN_STAGE,hypertune,modelling/hypertune.py)

lstm:
	$(call RUN_STAGE,lstm,modelling/lstm.py)

clean-logs:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '$(LOG_DIR)') { Remove-Item -Path '$(LOG_DIR)/*.log' -Force -ErrorAction SilentlyContinue }"

help:
	@echo Available targets:
	@echo   make features    - build gap-safe feature CSVs
	@echo   make baseline    - run local LightGBM validation
	@echo   make forecast    - refit all train rows and export forecast CSV
	@echo   make hypertune   - tune LightGBM with TimeSeriesSplit and export tuned forecast
	@echo   make lstm        - train PyTorch LSTM with VIF/correlation filters and export forecast
	@echo   make all         - run features, baseline, forecast
	@echo   make clean-logs  - remove logs/*.log
