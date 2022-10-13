import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
        title='Jobs to do',
        description='get_coword / make_graph / select_feature',
        dest='job')

# parser.add_argument('--job', type=str, default='make_graph')

# get coword
parser_get_coword = subparsers.add_parser('get_coword')
parser_get_coword.add_argument('--input', type=str, default='../_datasets/pre_patents.pkl')
parser_get_coword.add_argument('--output', type=str, default='../results')
parser_get_coword.add_argument('--not_fill_empty_time', default=True)
parser_get_coword.add_argument('--input_at_not_document_level', default=False)
parser_get_coword.add_argument('--no_count_multiplication',default=True)
parser_get_coword.add_argument('--timeline_normalization', default=False)
parser_get_coword.add_argument('--document_normalization', default=False)
parser_get_coword.add_argument('--sentence_normalization', default=False)

# make graph
parser_make_graph = subparsers.add_parser('make_graph')
parser_make_graph.add_argument('--centrality', default=['degree_centrality', 'betweenness_centrality', 'closeness_centrality'])
parser_make_graph.add_argument('--connectivity', default=['all_pairs_dijkstra'])
parser_make_graph.add_argument('--input', type=str, default='../results/coword_results.pkl')
parser_make_graph.add_argument('--output', type=str, default='../results/')




def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
