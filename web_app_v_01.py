# -*- coding: utf-8 -*-
# Веб-приложение ВКР: прогноз вероятности оттока клиента (Streamlit).
# Запуск:  streamlit run web_app_v_01.py
#
# История версий:
#   v0.1 — учебный шаблон курса (регрессия «соотношение матрица-наполнитель»,
#          модель model_regressor.sav + MinMaxScaler transformer.sav);
#          артефакты отсутствовали, входы без типизации — приложение
#          не запускалось.
#   v0.2 — адаптация под текущий проект ВКР «Прогнозирование оттока клиентов
#          при снижении качества оказания услуги (грузовые ж/д перевозки)»:
#          бинарная классификация, калиброванные вероятности (zua_model.pkl),
#          порог F1, рекомендация воздействия № 2140/р, SHAP-факторы.
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Порог F1 и диапазоны признаков — из пайплайна модели оттока
MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / 'models' / 'zua_model.pkl',
    Path(__file__).resolve().parent.parent
    / 'ВКР_предварительная_сдача' / '4_Приложение' / 'models' / 'zua_model.pkl',
    Path('models/zua_model.pkl'),
]
FEATURES_NUM = ['n_dispatches', 'wagons', 'share_on_time', 'avg_delay_days',
                'penalty_share', 'laid_over_count']
FEATURES_CAT = ['segment', 'size', 'scenario']


def find_model() -> Path:
    """Путь к артефакту модели: переменная окружения или каталоги поиска."""
    env = os.environ.get('ZUA_MODEL_PATH')
    if env and Path(env).exists():
        return Path(env)
    for cand in MODEL_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        'zua_model.pkl не найден. Положите файл в webapp/models/ либо '
        'задайте переменную окружения ZUA_MODEL_PATH. Искали: '
        + '; '.join(str(c) for c in MODEL_CANDIDATES))


@st.cache_resource(show_spinner='Загрузка модели…')
def load_model() -> dict:
    with find_model().open('rb') as f:
        return pickle.load(f)


def predict_churn(arts: dict, params: dict) -> float:
    """Калиброванная вероятность оттока в следующем месяце."""
    row = pd.DataFrame([params])
    return float(arts['pipeline'].predict_proba(row)[0, 1])


def shap_contributions(arts: dict, params: dict) -> pd.Series:
    """Топ-5 SHAP-вкладов признаков для одного клиента (по дереву)."""
    import shap  # опциональная зависимость

    tree_pipe = arts['tree_pipeline']
    pre = tree_pipe.named_steps['preprocessor']
    explainer = shap.TreeExplainer(tree_pipe.named_steps['model'])
    sv = explainer.shap_values(pre.transform(pd.DataFrame([params])))
    if isinstance(sv, list):          # старые версии shap: список по классам
        sv = sv[1]
    if getattr(sv, 'ndim', 2) == 3:   # новые версии: (n, k, 2)
        sv = sv[:, :, 1]
    names = [n.split('__')[-1] for n in pre.get_feature_names_out()]
    contrib = pd.Series(sv[0], index=names)
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:5]


def main():
    st.set_page_config(page_title='Прогноз оттока клиента', page_icon='🚂')
    st.title('Прогноз оттока клиента при снижении качества сервиса')
    st.caption('Модель ВКР: панель клиент×месяц (Agentklient.sim + sklearn); '
               'вероятности калиброваны (sigmoid, TimeSeriesSplit)')

    try:
        arts = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    threshold = arts['threshold']

    col1, col2 = st.columns(2)
    with col1:
        segment = st.selectbox('Сегмент клиента', ['loyal', 'sensitive', 'at_risk'],
                               help='loyal — наименее склонен к уходу, at_risk — наиболее')
        size = st.selectbox('Размер клиента', ['small', 'medium', 'large'])
        scenario = st.selectbox('Сценарий качества',
                                ['baseline', 'quality_decline'],
                                help='quality_decline — стресс сети: доля задержек 32%')
        n_dispatches = st.slider('Отправок за месяц', 0, 150, 12)
        wagons = st.slider('Вагонов за месяц', 0, 400, 30)
    with col2:
        share_on_time = st.slider('Доля отправок в срок', 0.0, 1.0, 0.85, 0.01)
        avg_delay_days = st.slider('Средняя просрочка, сут', 0.0, 5.0, 0.3, 0.01)
        penalty_share = st.slider('Доля пени в плате', 0.0, 0.5, 0.02, 0.005)
        laid_over_count = st.slider('Отставленных поездов за месяц', 0, 20, 1)

    params = {
        'n_dispatches': n_dispatches, 'wagons': wagons,
        'share_on_time': share_on_time, 'avg_delay_days': avg_delay_days,
        'penalty_share': penalty_share, 'laid_over_count': laid_over_count,
        'segment': segment, 'size': size, 'scenario': scenario,
    }

    if st.button('Predict', type='primary'):
        p = predict_churn(arts, params)
        st.metric('Вероятность оттока в следующем месяце', f'{p:.1%}')
        st.progress(min(p, 1.0))
        if p >= threshold:
            st.error(f'Высокий риск оттока (порог F1 = {threshold:.2f}): '
                     'рекомендуется корректирующее воздействие № 2140/р — '
                     'приоритетный подъём отставленных поездов клиента.')
        else:
            st.success('Риск оттока ниже порога реагирования.')

        st.divider()
        st.subheader('Факторы риска (SHAP)')
        try:
            top = shap_contributions(arts, params)
            st.bar_chart(top.sort_values())
            st.caption('Положительный вклад повышает риск оттока, '
                       'отрицательный — снижает (top-5 по модулю)')
        except ImportError:
            st.info('SHAP-факторы недоступны: пакет не установлен '
                    '(pip install shap).')
        except Exception as e:  # pragma: no cover
            st.info('SHAP-факторы недоступны: ' + str(e))


if __name__ == '__main__':
    main()
