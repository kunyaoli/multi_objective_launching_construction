import streamlit as st
import numpy as np
from pymoo.termination import get_termination
from joblib import load
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import plotly.express as px
import pandas as pd

st.title("顶推多目标优化（第一版计算工具演示）")
st.write("李焜耀")
st.write("likunyao@ccccltd.cn")
st.write("---")
Jacking_length = st.number_input(label="顶推距离/ 米", min_value=0.0, value=666.0)
L_unit_price = st.number_input(label="导梁单价/ 万元", min_value=0.0, value=0.15)
A_unit_price = st.number_input(label="临时墩单价/ 万元", min_value=1.0, value=201.75)

if st.button('运行'):
    st.write("In progress...")
    st.spinner(text="In progress...")


    class MyProblem(ElementwiseProblem):

        def __init__(self, Jacking_length, L_unit_price, A_unit_price):
            super().__init__(n_var=2,
                             n_obj=2,
                             n_ieq_constr=0,
                             xl=np.array([40.0, 0.6]),  # [320.0, 0.35, 0.0, 0.0, 0.25, 0.35, 2350.0, 42.5]
                             xu=np.array([80.0, 0.8]),  # [480.0, 1.0, 0.45, 0.65, 0.36, 0.45, 2450.0, 70.0]

                             )
            self.L_unit_price = L_unit_price
            self.A_unit_price = A_unit_price
            self.Jacking_length = Jacking_length
            self.Obj_displacement = r"./pred_model/objective_01_model.joblib"
            # self.Obj_stress = "./objective_02_model.joblib"

        def _evaluate(self, x, out, *args, **kwargs):
            # print(x)
            Var_displacement = load(self.Obj_displacement).predict(x.reshape(1, -1))[0]
            # Max_stress = load(self.Obj_stress).predict(x.reshape(1, -1))[0]

            A = np.floor(np.array(((self.Jacking_length - (0.6 * x[0] * x[1])) / x[0])))  # cal the number of A
            # print(A)
            L = np.floor(np.array(x[0] * x[1]))  # cal the length L
            # print(L)

            Cost = self.L_unit_price * L + self.A_unit_price * A

            out["F"] = [Var_displacement, Cost]


    algorithm = NSGA2(pop_size=20)
    termination = get_termination("n_gen", 5)

    problem = MyProblem(Jacking_length=Jacking_length, L_unit_price=L_unit_price, A_unit_price=A_unit_price)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)  # verbose=True

    X, F = res.opt.get("X", "F")

    Max_stress = load(r"./pred_model/objective_02_model.joblib").predict(X)

    df = pd.DataFrame()

    df['B'] = X[:, 0]
    df['p'] = X[:, 1]
    df['var'] = F[:, 0] * 1000
    df['cost'] = F[:, 1]
    df['max_stress'] = Max_stress

    fig = px.parallel_coordinates(df,
                                  dimensions=['B', 'p', 'var', 'cost', 'max_stress'],
                                  )

    fig.update_layout(
        margin=dict(l=20, r=40, t=40, b=20),  # 上下左右的边距大小
    )

    st.plotly_chart(fig, use_container_width=True)
    #st.plotly_chart(fig)
