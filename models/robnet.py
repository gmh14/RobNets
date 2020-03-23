from .basic_model import Network


def robnet(genotype_list, **kwargs):
    return Network(genotype_list=genotype_list, **kwargs)
