# 价格/政策数据（raw）

目的：把电价/气价/碳价等“外生经济参数”的来源与处理口径留痕，便于论文复现与答辩。

目录约定（建议）：

- `data/raw/price/electricity/`：电价取值（例如代理购电价格表、示例基准价）
- `data/raw/price/gas/`：天然气价格信息与换算假设
- `data/raw/price/carbon/`：碳市场政策与碳价参考

已补充可直接读取的 CSV（规则/参考值表）：
- `data/raw/price/catalog.csv`
- `data/raw/price/electricity/beijing_tou_2023_schedule.csv`
- `data/raw/price/electricity/beijing_tou_2023_multipliers.csv`
- `data/raw/price/electricity/beijing_proxy_purchase_electricity_flat_prices_2024-02.csv`
- `data/raw/price/gas/beijing_nonresidential_gas_prices_2024-2025.csv`
- `data/raw/price/gas/beijing_gas_powerplant_heat_ex_factory_price_2024-2025.csv`
- `data/raw/price/carbon/beijing_ets_reference_price.csv`

> 说明：本项目的“王牌外生数据文件”最终会把 `price_e/price_gas/carbon_tax` 写成逐 15min 的列；
> `data/raw/price/` 只负责记录“规则与来源”，不强制保存所有原始 PDF。
