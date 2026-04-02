import pandas as pd

analysis_df = pd.read_csv('data/analysis/initial_data.csv')
analysis_df = analysis_df[analysis_df['participant_tokens'] != 0]

# group by all columns except for rq_relevance, and maximize rq_relevance
grouping_cols = [col for col in analysis_df.columns if col != 'rq_relevance']
analysis_df = analysis_df.groupby(grouping_cols)['rq_relevance'].max().reset_index()

analysis_df['average_surprisal'] = analysis_df['total_surprisal'] / analysis_df['n_tokens']
analysis_df['project_id'] = analysis_df['source'].str.split('_').str[0]

# add a time variable that counts up from 1 for each source based on excerpt_id
analysis_df = analysis_df.sort_values(by=['source', 'excerpt_id'])
analysis_df['time'] = analysis_df.groupby('source').cumcount() + 1

# Create a new column called time_percent that divides the time variable by the max time for each source
analysis_df['time_percent'] = analysis_df.groupby('source')['time'].transform(lambda x: x / x.max())

# reorder columns to put project_id and time_percent first
new_order = [
    'excerpt_id', 'source', 'project_id', 'participant', 'excerpt', 'quality_criterion',  
    'clarity', 'immediate_relevance', 'specificity', 'attributed_meaning',
    'self_reportedness', 'spontaneity', 'rq_relevance', 'n_tokens', 'perplexity',
    'total_surprisal', 'average_surprisal', 'participant_tokens', 'interviewer_tokens',
    'token_ratio', 'intro_context', 'support_rapport', 'elaboration',
    'specifying', 'direct', 'indirect', 'structuring', 'interpreting',
    'clarification', 'time', 'time_percent'
]

analysis_df = analysis_df[new_order]


analysis_df.to_csv('data/analysis/final_data.csv', index=False)