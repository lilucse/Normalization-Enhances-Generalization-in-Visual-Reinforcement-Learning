from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.drq_cnsn import DrQ_cnsn
from algorithms.svea_cnsn import SVEA_cnsn
algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	'drq_cnsn':DrQ_cnsn,
	'svea_cnsn':SVEA_cnsn
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
