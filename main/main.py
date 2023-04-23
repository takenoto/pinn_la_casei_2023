# python -m main.main

from timeit import default_timer as timer

import numpy as np
import deepxde
import deepxde as dde 
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl


from domain.params.altiok_2006_params import (
    get_altiok2006_params,
    get_altiok2006_xp_data,
)
from domain.reactor.cstr_state import CSTRState
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.pinn_grid_search import run_pinn_grid_search
from main.numerical_methods import run_numerical_methods
from main.plot_xpsv import plot_xpsv, multiplot_xpsv, XPSVPlotArg
from main.plotting import *
from main.plotting.plot_pinn_3d_arg import PlotPINN3DArg
from main.plotting.pinn_conversor import get_ts_step_loss_as_xyz
from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults
from domain.params.solver_params import SolverParams
from domain.optimization.non_dim_scaler import NonDimScaler

from main.cases_to_try import *

from main.plotting.simple_color_bar import plot_simple_color_bar
from main.plotting.surface_3d import (
    plot_3d_surface,
    plot_3d_surface_ts_step_error,
    plot_3d_lines,
    PlotPINN3DArg,
)
from main.plotting.plot_lines_error_compare import plot_lines_error_compare


# For obtaining fully reproducible results
deepxde.config.set_random_seed(0)

xp_colors = ["#F2545B"]
"""
Cores apra dados experimentais
"""

pinn_colors = [
    "#7293A0",
    "#C2E812",
    "#ED9B40",
    "#B4869F",
    "#45B69C",
    "#FFE66D",
    "#E78F8E",
    "#861388",
    "#34312D",
]
"""
Lista de cores que representam diversos pinns
Ordem: RGBA
"""

num_colors = [
    "b",
    "#F39B6D",
    "#F0C987",
]
    

def main():
    deepxde.config.set_random_seed(0)

    plt.style.use("./main/plotting/plot_styles.mplstyle")

    run_cstr_nondim_test = True

    run_batch_nondim_test_v2 = True

    run_cstr_convergence_test = True

    run_fedbatch_cstr_nondim_test = False
    
    run_batch_ts_test = False

    run_case_6_check_layer_size = False
    
    run_case_6_ts_comparison_pinn_and_numeric = False
    
    run_compare_fedbatch_batch_and_cstr = True

    run_weights = False

   
    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)
    eq_params = altiok_models_to_run[0]
    # Serve pra batch e pra cstr
    initial_state = CSTRState(
        volume=np.array([5]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    initial_state_cstr = CSTRState(
        volume=np.array([1]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    # Serve pra fed-batch
    initial_state_fed_batch = CSTRState(
        volume=np.array([1]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    process_params_feed_off = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.0,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10.2,
        )

    process_params_feed_on = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=2, #L/h
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10.2,
        )

    process_params_feed_cstr = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=1, #L/h
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=24*4,
        )

    if run_cstr_nondim_test:

        print('RUN CSTR NEW NONDIM TEST')
        start_time = timer()
        cases = change_layer_fix_neurons_number(eq_params, process_params_feed_cstr)
        # cases = batch_tests_fixed_neurons_number(eq_params, process_params_feed_cstr)
        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            # estou assumindo que já começa no estado estacionário:
            return f_in_v*tf.math.pow(volume/max_reactor_volume, 7)
        pinns, p_best_index, p_best_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=cases
        )
        end_time = timer()
        print(f"elapsed time for BATCH NONDIM test = {end_time - start_time} secs")
        items = {}
        for i in range(len(pinns)):
            items[i + 1] = {
                "title": pinns[i].model_name,
                "cases": [
                    {"x": pinns[i].loss_history.steps, "y": np.sum(pinns[i].loss_history.loss_test, axis=1), "color": pinn_colors[0], "l": "-"},
                    {"x": pinns[i].loss_history.steps, "y": np.sum(pinns[i].loss_history.loss_train, axis=1), "color": pinn_colors[1], "l": "--"},
                ],
            }

        plot_comparer_multiple_grid(
            labels=['Loss (teste)', 'Loss (treino)'],
            figsize=(7.2*2, 8.2*2),
            gridspec_kw={"hspace": 0.35, "wspace": 0.14},
            yscale='log',
            sharey=True,
            sharex=True,
            nrows=2,
            ncols=3,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss",
        )

        def cstr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
            return f_in_v*pow(volume/max_reactor_volume, 7)    
        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc= cstr_f_out_calc_numeric,
            t_discretization_points=[240],
        )

        # PLOTAR O MELHOR DOS PINNS
        items = {}
        print(f'Pinn best index = {p_best_index}')
        print(f'Pinn best error = {p_best_error}')
       
        # Plotar todos os resultados, um a um
        num = num_results[0]
        for pinn in pinns:
            # x_pred = dde.geometry.TimeDomain(0, pinn.process_params.t_final / pinn.solver_params.non_dim_scaler.t_not_tensor)
            # prediction = pinn.model.predict(np.array([[0, 0.5, 1, 2, 4]]))
            # prediction = pinn.model.predict(np.vstack(np.ravel([0, 0.5, 1, 2, 4],)))
            pred_start_time = timer()
            prediction = pinn.model.predict(np.vstack(np.ravel(num.t*pinn.solver_params.non_dim_scaler.t_not_tensor,)))
            pred_end_time = timer()
            pred_time = pred_end_time - pred_start_time
            print(f'name = {pinn.model_name}')
            print(f'train time = {pinn.total_training_time} s')
            print(f'best loss test = {pinn.best_loss_test}')
            print(f'best loss train = {pinn.best_loss_train}')
            print(f'pred time = {pred_time} s')
            items = {}
            titles = ["X", "P", "S", "V"]
            # pinn_vals = [pinn.X, pinn.P, pinn.S, pinn.V]
            pinn_vals = [
                prediction[:, 0]*pinn.solver_params.non_dim_scaler.X_not_tensor,#pinn.X,
                prediction[:, 1]*pinn.solver_params.non_dim_scaler.P_not_tensor,#pinn.P,
                prediction[:, 2]*pinn.solver_params.non_dim_scaler.S_not_tensor,#pinn.S,
                prediction[:, 3]*pinn.solver_params.non_dim_scaler.V_not_tensor,#pinn.V]
            ]
            num_vals = [num.X,
            num.P,
            num.S,
            num.V]
            # TODO calcular erro l ABSOLUTO
            
            # Armazena os 4 erros
            error_L = []
            # Calcula os erros de X P S V
            for u in range(len(num_vals)):
                diff = np.subtract(pinn_vals[u], num_vals[u])
                total_error = 0
                # Pega ponto a ponto e soma o absoluto
                for value in diff:
                    total_error += abs(value)
                
                error_L.append(
                    total_error/len(pinn_vals[u])
                )
            print('ERROR XPSV')
            print(f'X = {error_L[0]}')
            print(f'P = {error_L[1]}')
            print(f'S = {error_L[2]}')
            print(f'V = {error_L[3]}')
            print(f'total = {np.sum(error_L)}')

            for i in range(4):
                items[i + 1] = {
                    "title": titles[i],
                    "cases": [
                        # Numeric
                        {"x": num.t, "y": num_vals[i], "color": pinn_colors[0], "l": "-"},
                        # PINN
                        {"x": num.t, "y": pinn_vals[i], "color": pinn_colors[1], "l": "--"},
                    ],
                }

            plot_comparer_multiple_grid(
                suptitle=pinn.model_name,
                labels=['Euler', 'PINN'],
                figsize=(6*1.5, 8*1.5),
                gridspec_kw={"hspace": 0.6, "wspace": 0.25},
                yscale='linear',
                sharey=False,
                nrows=2,
                ncols=2,
                items=items,
                title_for_each=True,
                supxlabel="tempo (h)",
                supylabel="g/L",
            )

        
        pass

    if run_batch_nondim_test_v2:
        print('RUN BATCH NEW NONDIM TEST')
        start_time = timer()
        cases = batch_nondim_v2(eq_params, process_params_feed_cstr)
        cases = batch_tests_fixed_neurons_number(eq_params, process_params_feed_cstr)
        cases = change_layer_fix_neurons_number(eq_params, process_params_feed_cstr)
        pinns, p_best_index, p_best_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases
        )
        end_time = timer()
        print(f"elapsed time for BATCH NONDIM test = {end_time - start_time} secs")
        items = {}
        for i in range(len(pinns)):
            items[i + 1] = {
                "title": pinns[i].model_name,
                "cases": [
                    {"x": pinns[i].loss_history.steps, "y": np.sum(pinns[i].loss_history.loss_test, axis=1), "color": pinn_colors[0], "l": "-"},
                    {"x": pinns[i].loss_history.steps, "y": np.sum(pinns[i].loss_history.loss_train, axis=1), "color": pinn_colors[1], "l": "--"},
                ],
            }

        plot_comparer_multiple_grid(
            labels=['Loss (teste)', 'Loss (treino)'],
            figsize=(7.2*2, 8.2*2),
            gridspec_kw={"hspace": 0.35, "wspace": 0.14},
            yscale='log',
            sharey=True,
            sharex=True,
            nrows=2,
            ncols=5,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss",
        )
            
        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc= lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[400],
        )

        # PLOTAR O MELHOR DOS PINNS
        items = {}
        print(f'Pinn best index = {p_best_index}')
        print(f'Pinn best error = {p_best_error}')
       
        # Plotar todos os resultados, um a um
        num = num_results[0]
        for pinn in pinns:
            # x_pred = dde.geometry.TimeDomain(0, pinn.process_params.t_final / pinn.solver_params.non_dim_scaler.t_not_tensor)
            # prediction = pinn.model.predict(np.array([[0, 0.5, 1, 2, 4]]))
            # prediction = pinn.model.predict(np.vstack(np.ravel([0, 0.5, 1, 2, 4],)))
            pred_start_time = timer()
            prediction = pinn.model.predict(np.vstack(np.ravel(num.t*pinn.solver_params.non_dim_scaler.t_not_tensor,)))
            pred_end_time = timer()
            pred_time = pred_end_time - pred_start_time
            print(f'name = {pinn.model_name}')
            print(f'train time = {pinn.total_training_time} s')
            print(f'best loss test = {pinn.best_loss_test}')
            print(f'best loss train = {pinn.best_loss_train}')
            print(f'pred time = {pred_time} s')
            items = {}
            titles = ["X", "P", "S", "V"]
            # pinn_vals = [pinn.X, pinn.P, pinn.S, pinn.V]
            pinn_vals = [
                prediction[:, 0]*pinn.solver_params.non_dim_scaler.X_not_tensor,#pinn.X,
                prediction[:, 1]*pinn.solver_params.non_dim_scaler.P_not_tensor,#pinn.P,
                prediction[:, 2]*pinn.solver_params.non_dim_scaler.S_not_tensor,#pinn.S,
                prediction[:, 3]*pinn.solver_params.non_dim_scaler.V_not_tensor,#pinn.V]
            ]
            num_vals = [num.X,
            num.P,
            num.S,
            num.V]
            for i in range(4):
                items[i + 1] = {
                    "title": titles[i],
                    "cases": [
                        # Numeric
                        {"x": num.t, "y": num_vals[i], "color": pinn_colors[0], "l": "-"},
                        # PINN
                        {"x": num.t, "y": pinn_vals[i], "color": pinn_colors[1], "l": "--"},
                    ],
                }

            plot_comparer_multiple_grid(
                suptitle=pinn.model_name,
                labels=['Euler', 'PINN'],
                figsize=(6*1.5, 8*1.5),
                gridspec_kw={"hspace": 0.6, "wspace": 0.25},
                yscale='linear',
                sharey=False,
                nrows=2,
                ncols=2,
                items=items,
                title_for_each=True,
                supxlabel="tempo (h)",
                supylabel="g/L",
            )

        
        pass

    if run_cstr_convergence_test:
        print('RUN CSTR NEW CONVERGENCE TEST')
        start_time = timer()
        cases = iterate_cstr_convergence(eq_params, process_params_feed_cstr)
        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            # estou assumindo que já começa no estado estacionário:
            return f_in_v*tf.math.pow(volume/max_reactor_volume, 7)
        pinns, p_best_index, p_best_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=cases
        )
        end_time = timer()
        print(f"elapsed time for CSTR CONVERGENCE test = {end_time - start_time} secs")
        items = {}
        for i in range(len(pinns)):
            print(i)
            items[i + 1] = {
                "title": pinns[i].model_name,
                "cases": [
                    {"x": pinns[i].loss_history.steps, "y": np.sum(pinns[i].loss_history.loss_test, axis=1), "color": pinn_colors[0], "l": "-"},
                    {"x": pinns[i].loss_history.steps, "y": np.sum(pinns[i].loss_history.loss_train, axis=1), "color": pinn_colors[1], "l": "--"},
                ],
            }

        plot_comparer_multiple_grid(
            labels=['Loss (teste)', 'Loss (treino)'],
            figsize=(7.2, 8.2),
            gridspec_kw={"hspace": 0.35, "wspace": 0.14},
            yscale='log',
            sharey=True,
            sharex=True,
            nrows=3,
            ncols=3,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss (test)",
        )

        # Cálculo numérico
        def cstr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
            return f_in_v*pow(volume/max_reactor_volume, 7)
            
        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc= cstr_f_out_calc_numeric,
            t_discretization_points=[240],
        )

        # PLOTAR O MELHOR DOS PINNS
        items = {}
        print(f'Pinn best index = {p_best_index}')
        print(f'Pinn best error = {p_best_error}')
       
        #  Plotar todos os resultados, um a um
        num = num_results[0]
        for pinn in pinns:
            items = {}
            titles = ["X", "P", "S", "V"]
            pinn_vals = [pinn.X, pinn.P, pinn.S, pinn.V]
            num_vals = [num.X, num.P, num.S, num.V]
            for i in range(4):
                items[i + 1] = {
                    "title": titles[i],
                    "cases": [
                        # Numeric
                        {"x": num.t, "y": num_vals[i], "color": pinn_colors[0], "l": "-"},
                        # PINN
                        {"x": pinn.t, "y": pinn_vals[i], "color": pinn_colors[1], "l": "--"},
                    ],
                }

            plot_comparer_multiple_grid(
                suptitle=pinn.model_name,
                labels=['Euler', 'PINN'],
                figsize=(6, 8),
                gridspec_kw={"hspace": 0.6, "wspace": 0.25},
                yscale='linear',
                sharey=False,
                nrows=4,
                ncols=1,
                items=items,
                title_for_each=True,
                supxlabel="time (h)",
                supylabel="g/L",
            )

        
        pass
    #---------------------------------------------------------------
    #---------------------------------------------------------------
    #---------------------------------------------------------------


    if run_fedbatch_cstr_nondim_test:
        print('RUN FED-BATCH + CSTR NON DIM TEST')
        # Testa valores de adimensinalização para cstr e fed batch e determina
        # em quais eles performam melhor
        cases_fed_batch = cases_non_dim(eq_params, process_params_feed_cstr)
        cases_cstr = cases_non_dim(eq_params, process_params_feed_cstr)

        start_time = timer()

        print('\n---------------\n')
        print('STARTING FED BATCH')
        # Fed batch
        p_fb, p_fb_i, p_fb_e = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_on,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_fed_batch
        )
        print('\n---------------\n')
        print('STARTING CSTR')
        # CSTR
        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            # estou assumindo que já começa no estado estacionário:
            return f_in_v*tf.math.pow(volume/max_reactor_volume, 7)
        p_cstr, p_cstr_i, p_cstr_e = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=cases_cstr)
        
        end_time = timer()
        
        # PLOTTING
        items = {}
        for i in range(len(p_fb)):
            items[i + 1] = {
                "title": p_fb[i].model_name,
                "cases": [
                    {"x": p_fb[i].loss_history.steps, "y": np.sum(p_fb[i].loss_history.loss_test, axis=1), "color": pinn_colors[2], "l": "--"},
                    {"x": p_cstr[i].loss_history.steps, "y": np.sum(p_cstr[i].loss_history.loss_test, axis=1), "color": pinn_colors[3], "l": ":"},
                ],
            }
    
        print(f"elapsed time for NONDIM test = {end_time - start_time} secs")

        plot_comparer_multiple_grid(
            labels=['Fed-Batch', 'CSTR'],
            figsize=(8.6, 6),
            gridspec_kw={"hspace": 0.25, "wspace": 0.11},
            yscale='log',
            sharey=True,
            sharex=True,
            nrows=2,
            ncols=3,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss (test)",
        )


        print('best index for fedbatch:')
        print(p_fb_i)
        print('best error for fedbatch:')
        print(p_fb_e)
        print('best index for cstr:')
        print(p_cstr_i)
        print('best error for cstr:')
        print(p_cstr_e)


    if run_weights:
        # DEMORA APROXIMADAMENTE 30 MINUTOS
        # Testa diferentes weights para batch, cstr e fedbatch
        # e plota os respectivos erros
        cases_weights = NEW_cases_to_try_WEIGHTS()

        start_time = timer()
        # Batch
        p_b, p_b_i, p_b_e = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_weights,
        )
        # Fed batch
        p_fb, p_fb_i, p_fb_e = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_on,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_weights
        )
        # CSTR
        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            # estou assumindo que já começa no estado estacionário:
            return f_in_v*tf.math.pow(volume/max_reactor_volume, 7)
        p_cstr, p_cstr_i, p_cstr_e = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=cases_weights)
        
        end_time = timer()
        
        # PLOTTING
        items = {}
        for i in range(len(p_b)):
            items[i + 1] = {
                "title": p_b[i].model_name,
                "cases": [
                    {"x": p_b[i].loss_history.steps, "y": np.sum(p_b[i].loss_history.loss_test, axis=1), "color": pinn_colors[1], "l": "-."},
                    {"x": p_fb[i].loss_history.steps, "y": np.sum(p_fb[i].loss_history.loss_test, axis=1), "color": pinn_colors[2], "l": "--"},
                    {"x": p_cstr[i].loss_history.steps, "y": np.sum(p_cstr[i].loss_history.loss_test, axis=1), "color": pinn_colors[3], "l": ":"},
                ],
            }

        plot_comparer_multiple_grid(
            labels=['Batch', 'Fed-Batch', 'CSTR'],
            figsize=(8.6, 6),
            gridspec_kw={"hspace": 0.25, "wspace": 0.11},
            yscale='log',
            sharey=True,
            sharex=True,
            nrows=3,
            ncols=3,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss (test)",
        )

        print(f"elapsed time for WEIGHT test = {end_time - start_time} secs")
        print('best index for batch:')
        print(p_b_i)
        print('best error for batch:')
        print(p_b_e)
        print('best index for fedbatch:')
        print(p_fb_i)
        print('best error for fedbatch:')
        print(p_fb_e)
        print('best index for cstr:')
        print(p_cstr_i)
        print('best error for cstr:')
        print(p_cstr_e)


    if run_compare_fedbatch_batch_and_cstr:
        print('RUN COMPARE FEDBATCH BATCH CSTR')
        # A diferença desse pros outros
        # É que ele usa o melhor weight, o melhor ts e o melhor layer_size, que foram determinados
        # separadamente
        # Conclusão
        # Se ele acertar o volume, normalmente erra o resto
        # Se ele errar o volume, necessariamente erra o resto pq dependem do volume
        # Ou seja, a variável mais sensível é aparentemente o volume

        # Plota X, P, S e V de batch, fed-batch e cstr
        # para V_S = 5L (volume do reator)
        # case_to_use = only_case_6_v3_for_ts(eq_params)
        case_to_use_b = only_case_6_v3_for_ts(eq_params)
        case_to_use_fb = only_case_6_v3_for_ts(eq_params)
        case_to_use_cstr = only_case_6_v3_for_ts(eq_params)
        # case_to_use['t_6']['adam_epochs'] = 10

        # NOVO Nº EPOCHS
        case_to_use_b['t_5']['adam_epochs'] = 800#00
        case_to_use_fb['t_5']['adam_epochs'] = 800#00
        case_to_use_cstr['t_5']['adam_epochs'] = 800#00
        case_to_use_b['t_5']['layer_size'] = [1] + [32] * 5 + [4]
        case_to_use_fb['t_5']['layer_size'] = [1] + [22] * 3 + [4]
        case_to_use_cstr['t_5']['layer_size'] = [1] + [22] * 2 + [4]

        case_to_use_b['t_5']['lbfgs_pre'] = True
        case_to_use_fb['t_5']['lbfgs_pre'] = True
        case_to_use_cstr['t_5']['lbfgs_pre'] = True
        case_to_use_b['t_5']['lbfgs_post'] = True
        case_to_use_fb['t_5']['lbfgs_post'] = True
        case_to_use_cstr['t_5']['lbfgs_post'] = True
        
        # BATCH
        # case_to_use_b['t_5']['w_X'] = 5
        # case_to_use_b['t_5']['w_V'] = 10
        
        # FED BATCH - melhor weight em 
        # # case_to_use_fb['t_5']['w_P'] = 3
        case_to_use_fb['t_5']['V_s'] = 5
        case_to_use_fb['t_5']['X_s'] = eq_params.Xm
        case_to_use_fb['t_5']['P_s'] = eq_params.Pm
        case_to_use_fb['t_5']['S_s'] = eq_params.So

        # CSTR - melhor weight em case 3
        case_to_use_cstr['t_5']['w_X'] = 3
        case_to_use_cstr['t_5']['V_s'] = 5
        case_to_use_cstr['t_5']['X_s'] = eq_params.Xm


        start_time = timer()
        pinns_batch, _, __ = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=case_to_use_b,
        )

        nums_batch = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[240],
        )

        pinns_fed_batch, _, __ = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_on,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=case_to_use_fb,
        )

        nums_fed_batch = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_on,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[240],
        )
        def cstr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
            # estou assumindo que já começa no estado estacionário:
            return f_in_v*pow(volume/max_reactor_volume, 7)

        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            # estou assumindo que já começa no estado estacionário:
            return f_in_v*tf.math.pow(volume/max_reactor_volume, 7)

        pinns_cstr, _, __ = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=case_to_use_cstr,)
           
        nums_cstr = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_numeric,
            t_discretization_points=[480],
        )
        end_time = timer()
        print(f"elapsed time for CSTR, BATCH & FEDBATCH XPSV = {end_time - start_time} secs")

        p_b = pinns_batch[0]
        n_b = nums_batch[0]
        p_fb = pinns_fed_batch[0]
        n_fb = nums_fed_batch[0]
        p_cstr = pinns_cstr[0]
        n_cstr = nums_cstr[0]

        items = {}
        # Em cada linha, todos de 1 bicho, ou seja, 4
        # São 12 cases no total
        # em cada mini gráfico, 2 valores de y: euler e pinn
        # col # Representa X P S V
        # row  # Representa BATCH FEDBATCH CSTR
        cols = 5 # Total
        cols = 4 # Isso remove a coluna de 'loss'
        rows = 3 # Total
        row_identifiers = ['Batch', 'Fed-batch', 'CSTR']
        column_identifiers = ['X', 'P', 'S', 'V', 'loss']
        p = [p_b, p_fb, p_cstr]
        n = [n_b, n_fb, n_cstr]
        # Vou usar a col "0" pra armazenar o tempo e 1 pra epochs, daí sempre use col+2
        num = [
            [n_b.t, n_b.X, n_b.P, n_b.S, n_b.V],
            [n_fb.t, n_fb.X, n_fb.P, n_fb.S, n_fb.V],
            [n_cstr.t, n_cstr.X, n_cstr.P, n_cstr.S, n_cstr.V],
        ]
        # Baseado em https://stackoverflow.com/questions/42142144/displaying-first-decimal-digit-in-scientific-notation-in-matplotlib
        # Ajustando casas decimais
        pinn  = [
            [p_b.t, p_b.loss_history.steps, p_b.X, p_b.P, p_b.S, p_b.V, np.sum(p_b.loss_history.loss_test, axis=1)],
            [p_fb.t, p_fb.loss_history.steps, p_fb.X, p_fb.P, p_fb.S, p_fb.V, np.sum(p_fb.loss_history.loss_test, axis=1)],
            [p_cstr.t, p_cstr.loss_history.steps, p_cstr.X, p_cstr.P, p_cstr.S, p_cstr.V, np.sum(p_cstr.loss_history.loss_test, axis=1)],
        ]
        for col in range(cols):
            for row in range(rows):
                i = (col%cols) + row*cols #o nº da row multiplica pelo nº de cols por rows
                items[i+1] = {
                    'ax_yscale': None if col <4 else 'log',
                    'y_label': 'g/L' if col < 3 else 'L' if col <4 else None,# 'loss (test)',
                    'y_majlocator': plt.MaxNLocator(3), #plt.LogLocator(numticks=20000) if col >=3 else None,
                    # 'y_minlocator': plt.LogLocator(subs='all', numticks=20000) if col >=3 else None,
                    # 'x_label': 'h' if col < 4 else 'epochs',
                    "title": f'{row_identifiers[row]} : {column_identifiers[col]}',
                    "cases": [],
                }
                if col <4: items[i+1]['cases'].append(
                        # Numeric
                        {"x": num[row][0], "y": num[row][col+1], "color": pinn_colors[0], "l": "-"}
                    )
                items[i+1]['cases'].append(
                        # PINN
                        {"x": pinn[row][0] if col < 4 else pinn[row][1], "y": pinn[row][col+2], "color": pinn_colors[1], "l": "--"},                        
                    )

        plot_comparer_multiple_grid(
            labels=['Euler','PINN'],
            figsize=(10, 6),
            gridspec_kw={"hspace": 0.5, "wspace": 0.55},
            yscale=None, #'linear',
            sharey=False,
            sharex=False,
            nrows=rows,
            ncols=cols,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="h",
            #supylabel="g/L",
        )

        plt.plot(p_b.loss_history.steps, np.sum(p_b.loss_history.loss_test, axis=1), linestyle='-', color=pinn_colors[3], label='Batch')
        plt.plot(p_fb.loss_history.steps, np.sum(p_fb.loss_history.loss_test, axis=1), linestyle='--', color=pinn_colors[4], label='Fed-Batch')
        plt.plot(p_cstr.loss_history.steps, np.sum(p_cstr.loss_history.loss_test, axis=1), linestyle=':', color=pinn_colors[2], label='CSTR')
        plt.xlabel('epochs')
        plt.ylabel('loss (test)')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.show()
        pass



        pass

    if run_case_6_check_layer_size:
        # NÃO É O CASE 6, É O MELHOR (NO CASO, 5!!!)
        print('LAYER SIZE TEST')
        # Checa a influência da layer_size para o t_s do case 6 -  EM BATCH
        # Case t6: comparando valores de erro para 9 redes, com e sem pré-otimização por lbfg-s
        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.0,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10.2,
        )


        pinns_cases_5 = iterate_layer_size_with_caset6(eq_params,
            process_params,
            use_lbfgs_pre=False,
            ts_case_num=5)

        start_time = timer()
        pinns_5, best_pinn_test_index_5, best_pin_test_error_5 = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=pinns_cases_5,
        )
        end_time = timer()
        print(f"elapsed batch t_best layers pinn grid time = {end_time - start_time} secs")


        # Agora itera e preenche o dict pra plotar
        # Prepara dict para plotar
        items = {}
        p5 = pinns_5
        for i in range(len(p5)):
            print(f'model = {p5[i].model_name}')
            print(f'best error for model = {np.sum(p5[i].best_loss_test)}')
            print('------------')
            items[i + 1] = {
                "title": p5[i].model_name,
                "cases": [
                    # PINN - case ts6
                    {"x": p5[i].loss_history.steps, "y": np.sum(p5[i].loss_history.loss_test, axis=1), "color": pinn_colors[1], "l": "-"},
                ],
            }

        plot_comparer_multiple_grid(
            # labels=['loss, case ts6', 'loss, case ts3'],
            figsize=(7.2, 8.2),
            gridspec_kw={"hspace": 0.35, "wspace": 0.14},
            yscale='log',
            sharey=True,
            sharex=True,
            nrows=4,
            ncols=3,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss (test)",
        )
        pass

    if run_case_6_ts_comparison_pinn_and_numeric:
        print('RUN COMPARE PINN NUMERIC XP')
        # Case t6: XPS comparando com euler e experimental
        # Só o caso 6, pedido pelo amaro - em BATCH
        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.0,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10.2,
        )
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=only_case_6_v3_for_ts(eq_params),
        )
        pinns = pinn_results

        xp_data = get_altiok2006_xp_data(2)

        num_results = run_numerical_methods(
            initial_state=initial_state,
            eq_params=eq_params,
            process_params=process_params,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            # Para ter as mesmas dimensões do pinn:
            t_discretization_points=[240],
        )

        title = "Concentrations over time for case 6: pinn, numeric (euler) and experimental"

        # Prepara dict para plotar
        items = {}
        pinn = pinn_results[0]
        num = num_results[0]
        titles = ["Cell", "Lactic Acid", "Lactose"]
        pinn_vals = [pinn.X, pinn.P, pinn.S]
        num_vals = [num.X, num.P, num.S]
        xp_vals = [xp_data.X, xp_data.P, xp_data.S]
        for i in range(3):
            items[i + 1] = {
                "title": titles[i],
                "cases": [
                    # Numeric
                    {"x": num.t, "y": num_vals[i], "color": pinn_colors[0], "l": "-"},
                    # PINN
                    {"x": pinn.t, "y": pinn_vals[i], "color": pinn_colors[1], "l": "--"},
                    # Experimental data
                    {
                        "x": xp_data.t,
                        "y": xp_vals[i],
                        "color": pinn_colors[2],
                        "l": "None",
                        'marker':'D'
                    },
                ],
                # Isso aqui n serve pra nada:
                # "x": pinn.loss_history.steps,
                # "y": [np.sum(pinn.loss_history.loss_test, axis=1)],
            }

        plot_comparer_multiple_grid(
            labels=['Euler', 'PINN', 'Experimental Data'],
            figsize=(6, 6),
            gridspec_kw={"hspace": 0.6, "wspace": 0.25},
            yscale='linear',
            sharey=False,
            nrows=3,
            ncols=1,
            items=items,
            suptitle=None,
            title_for_each=True,
            supxlabel="time (h)",
            supylabel="g/L",
        )
        pass

    if run_batch_ts_test:
        print('TS TEST')
        # Teste dos cases de t_s mantendo fixos layer_size, epochs e lbfgs
        # Plota loss por step
        """
        Teste da influência de t_s usando o reator batelada
        """
        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10.2,
        )

        start_time = timer()
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_to_try_batch_vary_ts(eq_params, process_params),
        )
        end_time = timer()
        print(f"elapsed batch T_S pinn grid time = {end_time - start_time} secs")

        # Cria dict pra plotar
        items = {}
        for p in range(len(pinn_results)):
            pinn = pinn_results[p]
            items[p + 1] = {
                "title": pinn.model_name,
                'cases':[
                    {
                        # To tentando fazer: de 1 a nº de steps
                        "x": pinn.loss_history.steps,
                        "y": np.sum(pinn.loss_history.loss_test, axis=1),
                        'l':'-',
                        "color": "tab:orange",
                    }
                ]
                
            }

        # Serão 6. Faremos 2 rows, 3 columns
        plot_comparer_multiple_grid(
            figsize=(7, 6),
            gridspec_kw={"hspace": 0.17, "wspace": 0.3},
            nrows=2,
            ncols=3,
            items=items,
            sharex=True,
            sharey=True,
            yscale='log',
            suptitle=None,
            title_for_each=True,
            supxlabel="epochs",
            supylabel="loss (test)",
        )
        pass


    print("--------------------")
    print("!!!!!!FINISED!!!!!!")
    print("--------------------")


if __name__ == "__main__":
    main()
