Avellaneda--Stoikov Market-Making Simulation

OVERVIEW :

This project implements and evaluates the Avellaneda--Stoikov (A--S)
market-making model.

The project focuses on two comparisons:

Symmetric vs. inventory-aware market making

Finite-horizon vs. infinite-horizon market making


OBJECTIVES :

Implement the Avellaneda--Stoikov market-making framework.

Implement inventory-aware reservation pricing.

Compare inventory-based and symmetric quoting.

Implement and compare finite- and infinite-horizon formulations.

Evaluate P&L, inventory risk, spreads, and order fills.

Use 300 Monte Carlo simulations for statistical comparison.

Visualize the resulting distributions and representative paths.

MATHEMATICAL MODEL

Mid-price

The mid-price follows a stochastic process of the form

dS_t = σ dW_t

where dS_t is the mid-price, σ is volatility, and dW_t is a
Wiener process.

Order-arrival intensity

Orders arrive according to an exponential intensity:

λ(δ) = A e^(-kδ)

where λ(δ) is the baseline arrival intensity, k controls the
sensitivity to quote distance, and δ is the quote's distance from
the mid-price.

Inventory-aware reservation price

The inventory strategy shifts the quote centre away from the mid-price:

r_t = S_t - inventory penalty

Positive inventory shifts the reservation price downward, encouraging
selling; negative inventory shifts it upward, encouraging buying.

Strategies

Symmetric strategy

Bid and ask quotes are centred around the current mid-price $S_t$.
Inventory is not explicitly used to shift the quote centre.

Inventory-based strategy

Bid and ask quotes are centred around the reservation price. This makes
the strategy respond to accumulated inventory.

Finite vs. Infinite Horizon

Finite horizon

The remaining time until the end of the trading period affects the
optimal quotes. As the terminal time approaches, inventory liquidation
becomes increasingly important.

Infinite horizon

The infinite-horizon formulation removes explicit dependence on
remaining time and uses a stationary inventory penalty controlled by
ω

In the current experiment, ω is derived from the chosen inventory
limit q_max.

The infinite-horizon formulation is useful when the market maker does
not have a fixed terminal liquidation time and a stationary quoting
policy is desired.

Simulation Parameters

Parameter                       Value

Initial price S_0                 100
Volatility σ                      2.0
Risk aversion γ                   0.1
Arrival parameter k               1.5
Arrival parameter A               140
Time step dt                      0.005
Initial inventory                 0
Horizon T                         1.0
Monte Carlo runs                  300
Random seed                        42
Infinite-horizon q_max            10

Monte Carlo Simulation

A single simulation represents one possible stochastic market path.
Because one path is not sufficient to evaluate a stochastic strategy,
the project runs 300 independent Monte Carlo simulations.

For each run, quantities such as final P&L, final inventory, inventory
risk, average spread, and number of fills are recorded. The resulting
distributions are then compared.

Evaluation Metrics

Final P&L: profit or loss at the end of the simulation.

P&L standard deviation: variability of final P&L.

Final inventory: inventory remaining at the end.

Inventory standard deviation: dispersion of final inventory.

Mean absolute inventory: typical inventory exposure.

Maximum absolute inventory: largest observed inventory exposure.

Average spread: average distance between bid and ask.

Order fills: number of buy, sell, and total executions.

Main Finite-vs-Infinite Results

For the 300-run experiment, the obtained results were:

Metric                    Finite Horizon   Infinite Horizon

Mean final P&L                     63.42              -2.27
Std. final P&L                      6.92               4.68
Mean final inventory               -0.19              -0.07
Std. final inventory                3.33               1.78
Mean inventory                      1.17               1.56
Maximum inventory                  11                  7
Average spread                      1.49               0.17
Mean buy fills                     49.27             111.26
Mean sell fills                    49.46             111.33
Mean total fills                   98.73             222.59

These results show a clear trade-off under the chosen parameters. The
finite-horizon strategy quotes more widely and receives fewer fills,
while producing substantially higher simulated P&L. The infinite-horizon
strategy quotes more aggressively, produces many more fills, and has
lower final-inventory dispersion, but its simulated P&L is substantially
lower.

These results are specific to the selected parameterisation and should
not be interpreted as proof that one horizon is universally superior.

Project Structure

avellaneda_stoikov/
├── market_models/
│   ├── arrivals.py
│   └── mid_price.py
├── simulator/
│   ├── agent.py
│   ├── engine.py
│   └── runner.py
├── strategy/
│   ├── inventory.py
│   ├── reservation.py
│   ├── reservation_infinite.py
│   ├── spread.py
│   └── spread_infinite.py
├── visualization/
│   └── dashboard.py
├── tests/
├── symmetric_vs_inventory.py
├── finite_vs_infinite_horizon.py
├── .gitignore
└── README.md

VISUALISATIONS : 

The project generates visualizations including:

Final P&L distributions

Final inventory distributions

Average spread comparison

Order-fill comparison

Inventory-risk comparison

Inventory over time

Inventory-risk evolution

Mid-price and bid/ask quotes

Mark-to-market P&L

SUMMARY : 

The project uses the Avellaneda--Stoikov framework as a simulation
platform to study how different pricing formulations affect
market-making performance.

The main comparisons are symmetric vs. inventory-aware pricing and
finite- vs. infinite-horizon pricing, with P&L, inventory exposure,
spreads, and execution frequency used to evaluate the strategies.
