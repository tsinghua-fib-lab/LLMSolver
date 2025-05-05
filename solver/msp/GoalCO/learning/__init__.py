"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from solver.msp.GoalCO.learning.tsp.decoding import decode as decode_tsp_like
from solver.msp.GoalCO.learning.pctsp.decoding import decode as decode_pctsp
from solver.msp.GoalCO.learning.cvrp.decoding import decode as decode_cvrp_like
from solver.msp.GoalCO.learning.op.decoding import decode as decode_op
from solver.msp.GoalCO.learning.cvrptw.decoding import decode as decode_cvrptw
from solver.msp.GoalCO.learning.mvc.decoding import decode as decode_mvc
from solver.msp.GoalCO.learning.mis.decoding import decode as decode_mis
from solver.msp.GoalCO.learning.kp.decoding import decode as decode_kp
from solver.msp.GoalCO.learning.upms.decoding import decode as decode_upms
from solver.msp.GoalCO.learning.jssp.decoding import decode as decode_jssp_like
from solver.msp.GoalCO.learning.mclp.decoding import decode as decode_mclp


decoding_fn = { "tsp": decode_tsp_like, "trp": decode_tsp_like, "sop": decode_tsp_like,
                "cvrp": decode_cvrp_like, "sdcvrp": decode_cvrp_like, "ocvrp": decode_cvrp_like, "dcvrp": decode_cvrp_like,
                "cvrptw": decode_cvrptw,
                "op": decode_op,
                "pctsp": decode_pctsp,
                "kp": decode_kp,
                "mvc": decode_mvc,
                "mis": decode_mis,
                "mclp": decode_mclp,
                "upms": decode_upms,
                "jssp": decode_jssp_like, "ossp": decode_jssp_like}

