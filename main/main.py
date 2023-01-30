# python -m main.main

from timeit import default_timer as timer

import numpy as np
import deepxde
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


def plot_compare_pinns_and_num(pinns, nums, eq_params, title=None):
    # --------------------------------------
    # PLOTTING
    # --------------------------------------

    # Plot simple color bar with ts, step, loss as x, y, z
    ts, step, loss = get_ts_step_loss_as_xyz(pinns=pinns)
    if False:
        plot_simple_color_bar(
            title="Time scaling factor, number of steps and error",
            x=ts,
            y=step,
            z=loss,
            x_label="t_s",
            y_label="step",
            z_label="loss",
        )

    # PLOTAR X
    multiplot_xpsv(
        title=title if title else "Concentrations over time for different methods",
        y_label="g/L",
        x_label="time (h)",
        t=[n.t for n in nums] + [pinn.t for pinn in pinns],
        X=[n.X for n in nums] + [pinn.X for pinn in pinns],
        P=None,
        S=None,
        V=None,
        y_lim=[0, eq_params.Xo * 10],
        # Essa linha só é interessante quando o erro é baixo...
        # err=[0 for n in nums] + [pinn.best_loss_test for pinn in pinns],
        scaler=[n.non_dim_scaler for n in nums]
        + [pinn.solver_params.non_dim_scaler for pinn in pinns],
        suffix=[n.model_name for n in nums] + [pinn.model_name for pinn in pinns],
        plot_args=[
            XPSVPlotArg(
                ls="-",
                color=num_colors[i % len(num_colors)],
                linewidth=7.0,
                alpha=0.25,
            )
            for i in range(len(nums))
        ]
        + [
            XPSVPlotArg(
                ls="--",
                color=pinn_colors[i % len(pinn_colors)],
                linewidth=4,
                alpha=1,
            )
            for i in range(len(pinns))
        ],
    )

    # PLOTAR V
    multiplot_xpsv(
        title=title if title else "Reactor content volume over time",
        y_label="L",
        x_label="time (h)",
        t=[n.t for n in nums] + [pinn.t for pinn in pinns],
        X=None,
        P=None,
        S=None,
        V=[n.V for n in nums] + [pinn.V for pinn in pinns],
        y_lim=[0, eq_params.Xo * 10],
        # Essa linha só é interessante quando o erro é baixo...
        # err=[0 for n in nums] + [pinn.best_loss_test for pinn in pinns],
        scaler=[n.non_dim_scaler for n in nums]
        + [pinn.solver_params.non_dim_scaler for pinn in pinns],
        suffix=[n.model_name for n in nums] + [pinn.model_name for pinn in pinns],
        plot_args=[
            XPSVPlotArg(
                ls="-",
                color=num_colors[i % len(num_colors)],
                linewidth=7.0,
                alpha=0.25,
            )
            for i in range(len(nums))
        ]
        + [
            XPSVPlotArg(
                ls="--",
                color=pinn_colors[i % len(pinn_colors)],
                linewidth=4,
                alpha=1,
            )
            for i in range(len(pinns))
        ],
    )

    if len(pinns) >= 1:
        plot_3d_surface_ts_step_error(pinns=pinns)

    if True:
        plot_lines_error_compare(pinns=pinns, group_by="t_s")


def main():

    plt.style.use("./main/plotting/plot_styles.mplstyle")

    run_batch_ts_test = False

    run_case_6_check_layer_size = False
    
    run_case_6_ts_comparison_pinn_and_numeric = False
    
    run_compare_fedbatch_batch_and_cstr = True

    run_weights = False

    run_cstr = False

    run_fed_batch = False

    run_batch = False

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
        case_to_use_b['t_5']['adam_epochs'] = 120000
        case_to_use_fb['t_5']['adam_epochs'] = 120000
        case_to_use_cstr['t_5']['adam_epochs'] = 120000
        case_to_use_b['t_5']['layer_size'] = [1] + [90] * 3 + [4]
        case_to_use_fb['t_5']['layer_size'] = [1] + [90] * 3 + [4]
        case_to_use_cstr['t_5']['layer_size'] = [1] + [90] * 3 + [4]

        # BATCH
        case_to_use_b['t_5']['w_X'] = 5
        case_to_use_b['t_5']['w_V'] = 10
        
        # FED BATCH - melhor weight em 
        case_to_use_fb['t_5']['w_P'] = 3

        # CSTR - melhor weight em case 3
        case_to_use_fb['t_5']['w_X'] = 3


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

    if run_cstr:
        initial_state = CSTRState(
            volume=np.array(1),  # antes era [1]
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )

        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.07,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=24,
        )

        def cstr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
            return f_in_v * (1 + (99 * volume / max_reactor_volume)) / 100
            if volume >= max_reactor_volume:
                return f_in_v
            else:
                return 0

        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            return f_in_v * (1 + (99 * volume / max_reactor_volume)) / 100
            # f_in_v = tf.keras.backend.get_value(f_in_v)
            # volume = tf.keras.backend.get_value(volume)
            # max_reactor_volume = tf.keras.backend.get_value(max_reactor_volume)
            return tf.cond(
                tf.greater_equal(
                    tf.cast(volume, tf.float32), tf.cast(max_reactor_volume, tf.float32)
                ),
                lambda: tf.cast(f_in_v, tf.float32),
                lambda: 0.0,
            )
            if volume >= max_reactor_volume:
                return f_in_v
            else:
                return 0

        # ----------------------------
        # NUMERICAL
        # ----------------------------
        num_results = run_numerical_methods(
            initial_state=initial_state,
            eq_params=eq_params,
            process_params=process_params,
            f_out_value_calc=cstr_f_out_calc_numeric,
        )
        pinn_results = []
        start_time = timer()
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=cases_to_try_vs_simplex(eq_params, process_params),
        )
        end_time = timer()
        print(f"elapsed cstr pinn grid time = {end_time - start_time} secs")
        # Plot
        plot_compare_pinns_and_num(
            title="Comparison of PINN and Numerical Results for CSTR varying ts",
            pinns=pinn_results,
            nums=num_results,
            eq_params=eq_params,
        )

    if run_fed_batch:
        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.07,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=24,  # 10,
        )
        # ----------------------------
        # NUMERICAL
        # ----------------------------
        num_results = run_numerical_methods(
            initial_state=initial_state,
            eq_params=eq_params,
            process_params=process_params,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
        )
        pinn_results = []
        start_time = timer()
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_to_try_ts(eq_params, process_params),
        )
        end_time = timer()
        print(f"elapsed fedbatch reactor pinn grid time = {end_time - start_time} secs")
        # Plot
        plot_compare_pinns_and_num(
            title="Comparison of PINN and Numerical Results for feedbatch reactor",
            pinns=pinn_results,
            nums=num_results,
            eq_params=eq_params,
        )

    if run_batch:
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
        # ----------------------------
        # NUMERICAL
        # ----------------------------
        num_results = run_numerical_methods(
            initial_state=initial_state,
            eq_params=eq_params,
            process_params=process_params,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
        )

        # ----------------------------
        # PINN
        # ----------------------------
        start_time = timer()
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_to_try_ts(eq_params, process_params),
        )
        end_time = timer()
        print(f"elapsed batch reactor pinn grid time = {end_time - start_time} secs")

        # Plot
        plot_compare_pinns_and_num(
            pinns=pinn_results, num=num_results, eq_params=eq_params
        )

    print("--------------------")
    print("!!!!!!FINISED!!!!!!")
    print("--------------------")


if __name__ == "__main__":
    main()
