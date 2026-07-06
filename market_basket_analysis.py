import streamlit as st,pandas as pd
from mlxtend.frequent_patterns import association_rules,fpgrowth


st.set_page_config(page_title="Market Basket Analysis", page_icon=":bar_chart:", layout="wide")
st.title("Market Basket Analysis")

hide_github_style = """
    <style>
    #GithubIcon {visibility: hidden;}
    button[title="View source code"] {display: none;}
    </style>
"""
st.markdown(hide_github_style, unsafe_allow_html=True)

@st.cache_data
def get_data():
    data = pd.read_csv("basket_analysis.csv",index_col=0)
    return data


@st.cache_data
def get_rules(frequent_itemsets):
    association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    association_rules_df['itemset'] = association_rules_df['antecedents'].apply(lambda x: ', '.join(list(x))) + ' -> ' + association_rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    association_rules_df[['already_bought','might_buy']]= association_rules_df['itemset'].str.split(' -> ', expand= True)
    association_rules_df['length_antecedents'] = association_rules_df['antecedents'].apply(lambda x: len(x))
    final_df = association_rules_df[['antecedents','itemset', 'support', 'confidence', 'lift', 'length_antecedents','already_bought','might_buy']]
    return final_df.sort_values(by='lift', ascending=False)
    

df=get_data()
common_patterns=fpgrowth(df, min_support=0.01, use_colnames=True)
rules =get_rules(common_patterns) 
with st.expander('View the original dataset'):
    st.dataframe(df)

st.write('Based on your selections, you can see the filtered patterns below. The table shows the patterns of items that are frequently bought together, along with their confidence scores. You can use this information to make informed decisions about product placement, promotions, and cross-selling strategies.')

if 'filter_data' not in st.session_state:
    st.session_state.filter_data = None

if 'filter_criteria' not in st.session_state:
    st.session_state.filter_criteria = None

if 'filter_df' not in st.session_state:
    st.session_state.filter_df = None

if 'item_count' not in st.session_state:
    st.session_state.item_count = None

if 'items' not in st.session_state:
    st.session_state.items = None

st.sidebar.header("Market Basket Analysis")
filter_data=st.sidebar.selectbox('How do you want to see the patterns?',options=['By Count of Items','By Items Selected'],index=0 )
st.session_state.filter_data = filter_data
if st.session_state.filter_data=='By Count of Items':
    item_count=st.sidebar.slider('Select the count of items that you bought', min_value=1, max_value=7, value=1)
    if st.sidebar.button('See Patterns By Number of Items'):
        st.session_state.filter_criteria = 'By Count of Items'
        st.session_state.item_count = item_count
        filtered_rules = rules[rules['length_antecedents'] == item_count]
        st.session_state.filter_df = filtered_rules[['itemset','confidence']].sort_values(by='confidence', ascending=False)
elif st.session_state.filter_data=='By Items Selected':
    items = st.sidebar.multiselect('Select Items', options=df.columns.tolist(),max_selections=6)
    strict_match = st.sidebar.checkbox("Match items exactly?",help="If checked, it will only show rules where the cart contains EXACTLY these items. If unchecked, it will find rules containing ANY of these items.")
    if st.sidebar.button('See Patterns By Selected Items'):
        st.session_state.filter_criteria = 'By Items Selected'
        st.session_state.item_count = None
        st.session_state.items = items.split(',') if isinstance(items, str) else items
        if items:
            if strict_match:
                filtered_rules = rules[rules['antecedents'].apply(lambda x: set(items) == x)]
                filtered_rules = filtered_rules.sort_values(by='confidence', ascending=False)
            else:
                filtered_rules = rules[rules['antecedents'].apply(lambda x: set(items).issubset(x))]
                filtered_rules = filtered_rules.sort_values(by='confidence', ascending=False)
            st.session_state.filter_df = filtered_rules[['already_bought','might_buy','confidence']].sort_values(by='confidence', ascending=False)


if st.session_state.filter_criteria and st.session_state.filter_df is not None:
    st.subheader(f"Filtered Patterns ({st.session_state.filter_criteria})")
    st.write(st.session_state.filter_df)

if st.session_state.filter_criteria=='By Count of Items' and st.session_state.filter_df is not None and st.session_state.item_count is not None:
    st.write('How to read the above table:')
    st.write(f'The table shows the patterns of items that are frequently bought together when you have selected {st.session_state.item_count} item(s). The "itemset" column represents the combination of items, and the "confidence" column indicates the likelihood of purchasing the "might_buy" item(s).')
    st.write('For example, if the itemset is "milk, bread -> butter" with a confidence of 0.8, it means that when customers buy milk and bread together, there is an 80% chance that they will also buy butter.')
elif st.session_state.filter_criteria=='By Items Selected' and st.session_state['items'] and st.session_state.filter_df is not None:
    st.write('### How to read the above table:')
    st.write(f'The table shows the patterns of items that are frequently bought together when you have selected the items: **{", ".join(st.session_state["items"])}**. The "already_bought" column represents the combination of items, and the "confidence" column indicates the likelihood of purchasing the "might_buy" item(s).')
    st.write('**Example:** If the already bought column is *"milk, bread"* and the might buy column is *"butter"* with a confidence of 0.8, it means that when customers buy milk and bread together, there is an 80% chance that they will also buy butter.')