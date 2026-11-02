import random

from poker.action import Action
from poker.card import Rank
from poker.game_state import GameState


# ---------------------------------------------------------------------------
# A fast rules-based bot. No simulation, so every decision takes microseconds.
#  - Preflop: a hardcoded table of all 169 starting hands, with their equity against
#    tight-to-loose ranges. Calls all-ins when equity beats the pot odds.
#  - Tracks how often each opponent goes all in, to estimate their shoving range.
#  - Postflop: classifies the made hand and draws, bets strong hands hard.
# Scoring is peak chips, so it accepts more all-in variance than normal poker.
# ---------------------------------------------------------------------------

RANKS = list(Rank)
VALUE = {rank: i + 2 for i, rank in enumerate(RANKS)}  # 2..14
CHEN_POINTS = {14: 10, 13: 8, 12: 7, 11: 6}
GAP_PENALTY = {0: 0, 1: 1, 2: 2, 3: 4}

# Every starting hand: (percentile among all 1326 combos, strongest first; equity % vs top 3/6/10/15/20/30/40/55/75/100% ranges)
# Precomputed offline with 20k/4k Monte Carlo deals per entry.
RANGE_PCTS = [3, 6, 10, 15, 20, 30, 40, 55, 75, 100]
HANDS = {
    'AA': (0.5, (80.5, 82.4, 84.8, 86, 86.3, 86.3, 86.3, 86, 85.5, 85.7)),
    'KK': (0.9, (68, 70.6, 72.1, 72.7, 73, 77.1, 77.2, 79.7, 81.9, 82)),
    'QQ': (1.4, (54.9, 62.5, 65.4, 67.3, 68.4, 71, 71.8, 76.3, 78, 80.1)),
    'JJ': (1.8, (44.4, 56, 60.2, 62.1, 65.6, 68.6, 68.5, 70.4, 74.7, 78.2)),
    'TT': (2.3, (30.3, 49.5, 54.9, 60, 61.6, 63, 66.7, 68.1, 71.1, 74.3)),
    '99': (2.7, (19.5, 41.9, 49.1, 53.8, 55.5, 59, 62.4, 64.8, 67.6, 71.7)),
    '88': (3.2, (19.2, 36.4, 47.9, 50.7, 53.2, 57.1, 57.8, 61.7, 64.2, 68.6)),
    'AKs': (3.5, (42.2, 47.9, 58.6, 61.3, 65.1, 66.8, 65.8, 68.1, 67.7, 67)),
    'AQs': (3.8, (39.5, 40.9, 51.7, 56.3, 59.7, 61.4, 63.8, 65.3, 65.5, 65.1)),
    '77': (4.2, (19.7, 30.9, 43.7, 48.5, 50.3, 53.9, 56.2, 58.1, 61, 66.1)),
    'AK': (5.1, (38.2, 44.1, 57.5, 60.2, 63.6, 65.8, 65.2, 65.6, 65.8, 64.3)),
    'AJs': (5.4, (36.4, 37.8, 45.9, 52.1, 56.1, 60.7, 63.4, 64, 65.6, 66.1)),
    'AQ': (6.3, (37, 38.7, 49.4, 54.9, 57.2, 61.8, 62.2, 63.4, 64.8, 62.9)),
    'ATs': (6.6, (34.5, 36.4, 39.6, 48.9, 54, 58.1, 61, 60.1, 63.4, 64.7)),
    'AJ': (7.5, (32.8, 33.3, 43.4, 50.6, 54.9, 58.4, 61.3, 60.8, 62.8, 63.9)),
    'KQs': (7.8, (38, 37.4, 39.8, 42.8, 48.5, 51.5, 56.1, 60, 63, 63.8)),
    'A9s': (8.1, (31.5, 32.2, 34.5, 44.4, 49.2, 54.4, 56.9, 59.4, 61.6, 62.5)),
    'AT': (9.0, (28.7, 31.4, 35.3, 45, 50.1, 55.6, 58, 59, 63.1, 62.5)),
    '66': (9.5, (19.6, 30.4, 40.1, 45.9, 47.5, 50.8, 54.5, 55.8, 58.6, 65.2)),
    'KJs': (9.8, (34.3, 35.4, 37.5, 41.6, 45.6, 50.3, 52.9, 58.1, 60.5, 63.3)),
    'A8s': (10.1, (29.6, 31.7, 34.6, 39.3, 46.8, 52.7, 56.3, 57.8, 60.1, 63.9)),
    'KTs': (10.4, (30.8, 34.3, 36.5, 39.1, 42.4, 47.1, 52.8, 56.1, 59.2, 61)),
    'KQ': (11.3, (32.6, 33.1, 35.8, 39.8, 46.2, 48.6, 51.6, 56.7, 59.6, 61.9)),
    'QJs': (11.6, (31.9, 36.9, 37, 38.9, 40.3, 45.9, 47, 53.2, 56.9, 60.3)),
    'A7s': (11.9, (29.2, 31, 34, 39.8, 44.9, 49.4, 53, 57.4, 57.8, 61.1)),
    'A8': (12.8, (25.9, 27.1, 31.2, 36.9, 42.6, 49.1, 54.4, 55.5, 58, 60.3)),
    '55': (13.3, (19.9, 31, 38.9, 42.3, 46.2, 49.9, 51, 54.1, 57, 60.6)),
    'A5s': (13.6, (31, 30.6, 34.5, 37.7, 42.6, 47.7, 51.2, 54.3, 56.6, 59.3)),
    'A9': (14.5, (24.4, 29.4, 31.3, 40.5, 46.2, 52, 54.8, 57, 58.4, 60.1)),
    'KJ': (15.4, (30.1, 31.7, 34.5, 38, 41.3, 48.3, 50.9, 56, 58.6, 61)),
    'K9s': (15.7, (28, 31.5, 36.3, 36.6, 40.7, 45.2, 49.7, 55, 57.4, 59.1)),
    'KT': (16.6, (27.3, 31.2, 33, 37.7, 39, 46.7, 48.8, 53.4, 57, 59)),
    'A6s': (16.9, (29.3, 30.4, 32.8, 36.2, 43.5, 47.3, 52.7, 55.2, 57, 60.5)),
    'A4s': (17.2, (30.9, 32.2, 33.8, 38.7, 42.3, 47.2, 50.6, 53.8, 57.1, 57.9)),
    'QTs': (17.5, (29.5, 34.7, 36.1, 38.9, 38.8, 44.9, 46.2, 51.4, 56.3, 60)),
    'A7': (18.4, (27.1, 26, 29.7, 35.4, 42, 47.2, 51.4, 52.9, 56.6, 58.3)),
    'K9': (19.3, (23.7, 29, 32.6, 33.1, 34.8, 42.8, 45.8, 51, 54.4, 58.7)),
    'K8s': (19.6, (27, 31.1, 33.5, 34.8, 35.4, 41.9, 47.2, 50.9, 54.8, 58.4)),
    'A3s': (19.9, (32, 31.6, 34.8, 38.5, 43.8, 45, 49.5, 53.1, 54.5, 58.2)),
    'QJ': (20.8, (28.3, 31.9, 33, 35.1, 37.4, 41.9, 45.5, 50.8, 56.4, 57.9)),
    'Q9s': (21.1, (25.6, 34.1, 35.1, 37.3, 37.6, 40.6, 44, 49.5, 53.6, 58.2)),
    'QT': (22.0, (25.3, 32.6, 33.4, 36.1, 36.6, 39.2, 43.5, 49.5, 53.4, 58.1)),
    'A6': (22.9, (26.4, 27.9, 28, 32.5, 39.2, 44.9, 49.9, 53.2, 55.1, 58.4)),
    'A5': (23.8, (28, 27, 30.2, 34.7, 39.4, 46.5, 49, 52.4, 54, 57.2)),
    'K7s': (24.1, (25.8, 29.7, 33, 35.6, 34.6, 41.2, 45.8, 50.6, 54.7, 58)),
    'JTs': (24.4, (27.3, 36.3, 36.2, 40.4, 39.9, 40.1, 45.1, 48.1, 52.5, 57.6)),
    '44': (24.9, (18.8, 29.8, 37.9, 40.7, 45, 47, 50.4, 51.1, 53.3, 57)),
    'K6s': (25.2, (28.7, 29, 32.9, 35.7, 37.9, 39.2, 44, 48.7, 53.5, 56.5)),
    'A4': (26.1, (26.7, 29.4, 29.8, 33.4, 39.6, 43.6, 48, 50.7, 53.7, 56.4)),
    'A2s': (26.4, (29.3, 32.4, 32.5, 36.8, 42.6, 46.5, 49, 52.8, 54.6, 55.6)),
    'K8': (27.3, (23.8, 26.6, 30.2, 31.3, 34.1, 38.4, 44.5, 49, 53.5, 56.3)),
    'Q8s': (27.6, (24.9, 31.2, 33.1, 34.7, 35.7, 38, 41.1, 45.7, 51.9, 57.1)),
    'K5s': (27.9, (27.1, 30.1, 33, 35.5, 35.2, 39.4, 43.6, 47.3, 52.3, 56)),
    'Q9': (28.8, (22.1, 29.8, 32.2, 32.6, 33.3, 35.6, 41.6, 45.4, 50.8, 55.4)),
    'J9s': (29.1, (24.4, 33.1, 34.3, 36.4, 38.2, 38, 40.4, 44.3, 50.9, 55)),
    'K7': (30.0, (23.8, 25.6, 30.5, 32.9, 31, 37.9, 42.1, 47.1, 52.9, 54.1)),
    'A3': (30.9, (27.5, 26.6, 28.6, 34, 38.9, 43.2, 47.5, 52, 53.2, 54.7)),
    'JT': (31.8, (23.6, 32.7, 34.9, 36.3, 37.2, 39, 41.6, 44.5, 51.3, 54.9)),
    'A2': (32.7, (26.7, 27.3, 30.1, 32.8, 39.2, 44.3, 46.8, 49.5, 51.9, 55.2)),
    'K4s': (33.0, (27.1, 29.6, 33.1, 33, 34.3, 38.9, 42.1, 46.5, 50.9, 55.2)),
    'K6': (33.9, (23.5, 25.9, 29.3, 30.6, 32.7, 36.1, 40.2, 46.4, 51.2, 54.8)),
    'T9s': (34.2, (21.6, 33.4, 35, 35.9, 38, 40, 40.3, 44.4, 47.8, 53.8)),
    '33': (34.7, (18.7, 29.1, 38.5, 41.4, 44.1, 46.9, 47.8, 50.3, 51.8, 54.9)),
    'Q7s': (35.0, (24.3, 28.4, 32.1, 34.3, 34.8, 36.1, 39.7, 44.7, 49.5, 53.9)),
    'K3s': (35.3, (25.4, 30.1, 32.5, 34.3, 36, 38.3, 40.7, 47.3, 51.9, 53.2)),
    'J8s': (35.6, (23.2, 31, 34.7, 35.4, 35.8, 37.9, 39.2, 43.4, 48.4, 54.3)),
    'Q8': (36.5, (21.3, 27.4, 30.5, 31.9, 31.8, 35.1, 37.5, 43.1, 49.5, 53)),
    'J9': (37.4, (19.5, 28, 31.1, 34, 36, 36.1, 37.6, 43.3, 48.7, 52.1)),
    'K5': (38.3, (23.5, 26.1, 29.3, 31.2, 31, 36.5, 38.7, 43.5, 49.5, 53)),
    'Q5s': (38.6, (23.2, 28.6, 32.7, 34.2, 35, 37.5, 38.5, 43.1, 49, 53.1)),
    'K2s': (38.9, (26.3, 28.3, 32.1, 35, 33.9, 38.1, 41.6, 45.6, 50.6, 53.4)),
    'Q6s': (39.2, (24.8, 29, 31.9, 34.8, 36, 37, 40.1, 43.1, 48.1, 53.3)),
    'J7s': (39.5, (22.9, 28.8, 32.5, 35.6, 35.5, 37.1, 40.5, 40.7, 46.6, 52.5)),
    'T8s': (39.8, (22.1, 30.6, 33.3, 36.2, 37.1, 38.7, 39.5, 41.2, 47.7, 51.5)),
    'K4': (40.7, (23.6, 24.9, 29.5, 29.9, 31.4, 34.4, 38.6, 43.7, 49.4, 54)),
    'Q4s': (41.0, (24.1, 29.7, 31.8, 32.5, 34.1, 34.7, 37.7, 41.7, 48, 51.8)),
    'T9': (41.9, (18.1, 30.7, 32.3, 33.9, 35.6, 35.7, 39, 39.9, 45.9, 49.8)),
    'Q6': (42.8, (20.6, 26.3, 28.2, 30, 30.9, 33.7, 36, 39.9, 47.4, 50.6)),
    'K3': (43.7, (21.7, 25.4, 29.9, 30.5, 31.1, 36.7, 39.9, 42.8, 46.8, 49.6)),
    'Q7': (44.6, (20.2, 25.4, 28.7, 30.4, 31.1, 32.3, 36.5, 41.3, 47.3, 51.9)),
    'Q5': (45.6, (21.3, 26.2, 29.7, 30.1, 31.1, 32, 34.3, 38.9, 44.9, 50.2)),
    'J8': (46.5, (19.5, 27.3, 30.6, 31.7, 33.1, 36.2, 37.3, 40.5, 46.1, 51.5)),
    'J6s': (46.8, (21.9, 28.2, 31.2, 34, 34.9, 34.9, 37.1, 39, 45.3, 49.6)),
    'Q3s': (47.1, (23.8, 28.2, 31.6, 32.7, 35.1, 35.1, 36.5, 40.5, 47.9, 51.1)),
    '98s': (47.4, (20.1, 30.2, 36.6, 36.2, 37.6, 38.9, 39.5, 41.5, 45.4, 51.4)),
    'T7s': (47.7, (21.7, 30.1, 33.8, 34.1, 33.8, 36.5, 39, 40.6, 44.4, 51.9)),
    'J5s': (48.0, (22.5, 28, 31.4, 33.5, 33.1, 34.7, 35.2, 38.2, 44.5, 49.2)),
    '22': (48.4, (19, 30.3, 38, 41, 44.4, 45.6, 46.6, 49.2, 49.1, 52.2)),
    'Q2s': (48.7, (23, 28.8, 32.1, 31.8, 34.4, 35.5, 37.6, 41.3, 46.2, 49.2)),
    'K2': (49.6, (22, 24.3, 28.4, 28.1, 29.7, 34.8, 37.2, 42.3, 47, 50.7)),
    'J7': (50.5, (17.5, 25.2, 29.8, 32.6, 32.9, 32.2, 34.3, 37, 44.8, 49.5)),
    'T8': (51.4, (17.7, 27.4, 29.9, 32.1, 33.1, 34.2, 35.5, 38.3, 44.8, 48.8)),
    'J4s': (51.7, (21.3, 27, 29.9, 32.6, 33.5, 34.2, 35.3, 38.1, 43.8, 47.4)),
    '97s': (52.0, (19.8, 28.5, 33.3, 36.2, 37.4, 37.8, 38.8, 40.3, 42.8, 49.5)),
    'T6s': (52.3, (20.3, 29.2, 30.1, 34.1, 34.5, 35.9, 37, 39.1, 42.7, 48.5)),
    'J3s': (52.6, (22, 27.8, 29.5, 30.5, 33.3, 34.3, 36.2, 37.3, 44, 47.6)),
    'Q4': (53.5, (18.7, 26.5, 27.5, 29, 30.9, 31.8, 33.5, 38.8, 44.4, 48.2)),
    'J6': (54.4, (17.7, 23.2, 26.8, 29.1, 30.6, 32.8, 33.3, 37.3, 41.2, 49.3)),
    'Q3': (55.4, (20.1, 24.3, 27.7, 31.5, 32, 30.3, 33.9, 37.4, 44.3, 48.3)),
    '98': (56.3, (15.6, 26.4, 32.1, 33, 33.9, 36.5, 37.7, 37.6, 42, 48.3)),
    '87s': (56.6, (20, 29.4, 34.1, 35.7, 36.2, 39, 39.3, 40.6, 41.9, 48)),
    '96s': (56.9, (18, 28.9, 31.4, 33.9, 35.1, 36.2, 38.6, 39.5, 42, 47.4)),
    'Q2': (57.8, (19.8, 23.8, 28.3, 28.3, 31, 29.3, 34.3, 38.3, 42.7, 47.2)),
    'J2s': (58.1, (20.1, 27.6, 30, 32.3, 33.8, 34.3, 36, 38.4, 42.8, 47.9)),
    'T7': (59.0, (16.6, 24.9, 28.9, 31.8, 32.7, 32.6, 33.9, 36.5, 41.8, 49)),
    'J5': (59.9, (17.9, 24.2, 27.5, 29, 30.6, 31.1, 33.4, 35, 40.4, 45.9)),
    'T5s': (60.2, (18.9, 26.4, 30.5, 32.1, 34.1, 35, 34.6, 37, 41.4, 46.9)),
    'T4s': (60.5, (20, 26.9, 29.1, 31.2, 32.6, 34.5, 36.5, 37.8, 40, 46.8)),
    '86s': (60.8, (20.6, 25.8, 33.3, 34.4, 36.3, 38, 38.2, 39.7, 41.4, 46.4)),
    'J4': (61.7, (17.7, 24.3, 27, 29.7, 28.9, 30.7, 33.9, 34.2, 42.4, 46.2)),
    'T3s': (62.0, (18.6, 26.9, 30.1, 31.9, 32.6, 35, 35.3, 35.9, 38.5, 46.9)),
    'T6': (62.9, (16.5, 24.9, 26.9, 29.1, 30.4, 32.5, 34.6, 34.5, 39.3, 46.2)),
    '76s': (63.2, (21.4, 28.2, 34.6, 34.8, 36.7, 39, 39.1, 39.3, 41.5, 46.1)),
    '95s': (63.5, (18.6, 27.5, 31.3, 32.5, 33.9, 36, 36.8, 36.4, 39.5, 47)),
    '97': (64.4, (16.5, 24.5, 30.8, 31.5, 33.9, 33.9, 35.7, 35.9, 40.5, 45.9)),
    '87': (65.3, (16.5, 24.3, 30.6, 33.1, 33.8, 36.2, 36.2, 38.5, 40.1, 45.3)),
    'J3': (66.2, (17.2, 23.5, 25.1, 29.3, 29.3, 30.3, 31.4, 35.4, 41.8, 46)),
    'T2s': (66.5, (18.6, 27.4, 29, 31.6, 31.8, 33.3, 35.4, 36.3, 38.4, 44)),
    '94s': (66.8, (16, 26.1, 29.2, 31.8, 32, 33.4, 35.3, 34.9, 36.5, 44.9)),
    'J2': (67.7, (16.4, 22.6, 25.3, 30.1, 29.5, 30.8, 30.6, 33.3, 39.5, 44.7)),
    '85s': (68.0, (19.6, 26.2, 31.7, 32, 35.2, 37.1, 37.1, 37.7, 39.9, 45.5)),
    '75s': (68.3, (21.7, 26.4, 32.3, 32.6, 35.6, 37.8, 36.2, 38.4, 38.7, 43.6)),
    'T5': (69.2, (14.8, 22.9, 26.8, 28.6, 31.3, 31, 31.5, 33.3, 37.5, 44.9)),
    '96': (70.1, (15.2, 24.1, 28.1, 30.6, 32.2, 33.3, 34.1, 35, 39.7, 44.2)),
    'T4': (71.0, (14.2, 23.4, 25.5, 29.4, 29.6, 31.2, 31, 34.3, 37.9, 44.8)),
    '65s': (71.3, (22.2, 27.6, 32.6, 35.3, 36.1, 37.8, 38.3, 38, 41.7, 43.3)),
    '86': (72.2, (17, 23.8, 29.2, 31.4, 32.1, 34.3, 35.1, 35.5, 38.3, 43.3)),
    '93s': (72.5, (16, 25.6, 29.5, 30.6, 32.6, 34.8, 34.3, 35.5, 38.7, 42.2)),
    '84s': (72.9, (17.1, 25.1, 30.8, 31.3, 35.1, 33.5, 34.9, 36.5, 37.1, 42.6)),
    '92s': (73.2, (15.1, 25.5, 28, 31.8, 32.2, 33.9, 35.3, 34.9, 36.8, 43.3)),
    'T3': (74.1, (14, 22.8, 26.4, 28.4, 29, 30.3, 32.1, 32.3, 36.5, 43.8)),
    '76': (75.0, (16.9, 23.2, 28, 32.4, 33.6, 35.3, 35.9, 36.1, 37.4, 42.3)),
    '95': (75.9, (14.2, 22.7, 27.5, 28.9, 31, 31.7, 33, 33.5, 37.4, 42.5)),
    'T2': (76.8, (14, 22.4, 25.4, 27.6, 30.5, 29.6, 30.3, 31.9, 35.3, 40.6)),
    '85': (77.7, (15.2, 22.4, 29, 28.6, 31.9, 32.6, 33.8, 33.8, 36.3, 42.2)),
    '83s': (78.0, (16.8, 24.8, 29.2, 30.5, 33.6, 32.9, 34.8, 35.9, 35.9, 41.3)),
    '74s': (78.3, (18.5, 24.7, 31.2, 31.9, 33.1, 34.5, 36.9, 37, 38.6, 42.3)),
    '54s': (78.6, (22, 26.8, 31.2, 33.9, 36.3, 37.6, 38.2, 38.8, 39, 41.4)),
    '64s': (78.9, (21.2, 27.1, 31.5, 33.1, 34.2, 35.5, 37.6, 36.4, 38.5, 42.1)),
    '75': (79.8, (17.1, 22.4, 28.6, 31.6, 31.7, 32, 34.4, 35.5, 35.4, 39.6)),
    '93': (80.7, (11.8, 22.8, 24.8, 28, 28.2, 30.1, 30.8, 31.4, 33.1, 40.1)),
    '94': (81.6, (12.5, 21.8, 25, 27.5, 28.8, 30.1, 30.9, 31.8, 35.4, 39.4)),
    '65': (82.5, (18, 22.9, 29.5, 32.8, 33, 34.9, 35.8, 34.9, 37.2, 40.2)),
    '73s': (82.8, (17.3, 24.6, 29.9, 29.8, 32.7, 32.7, 35.8, 35.6, 35.8, 40)),
    '82s': (83.1, (15.7, 23.8, 29.2, 29.6, 33.3, 33.7, 33.6, 34.3, 34.8, 41.2)),
    '63s': (83.4, (19.1, 25.1, 29.6, 31.4, 34.4, 34.7, 35, 36.5, 36.7, 38.8)),
    '53s': (83.7, (20.7, 26.2, 31.4, 32, 32.9, 34.3, 35.8, 36.9, 37.1, 38.7)),
    '92': (84.6, (11.8, 21.1, 25.4, 27.4, 28.8, 28.6, 29.2, 31.6, 33.8, 38.5)),
    '84': (85.5, (13.9, 22.2, 27.2, 28.1, 29.8, 30.2, 31.5, 32.6, 33.9, 40.7)),
    '43s': (85.8, (20.7, 26.3, 31, 32, 34.5, 33.9, 35.4, 36.9, 37, 39.1)),
    '74': (86.7, (14.7, 20.4, 27.8, 30.2, 30, 31.3, 33.1, 33.5, 35.3, 39)),
    '72s': (87.0, (15.8, 21.8, 28.8, 29.7, 30.4, 32.9, 33.4, 34.2, 35.3, 38.3)),
    '52s': (87.3, (19.4, 26.1, 29.8, 32.4, 33.4, 34.2, 35.1, 35.8, 37.3, 37.9)),
    '64': (88.2, (16.6, 23.9, 27.6, 31, 30.6, 32.5, 32.7, 33.1, 35.2, 39.2)),
    '54': (89.1, (19.3, 24.7, 29.1, 29.4, 32.2, 33.8, 35.8, 35.2, 34.8, 36.6)),
    '62s': (89.4, (18.8, 24.6, 29.2, 31.3, 34, 34.2, 34.4, 34.8, 34.8, 35.8)),
    '83': (90.3, (11.6, 20.3, 25.4, 27, 29.2, 30.1, 31.3, 31.6, 33, 37.4)),
    '42s': (90.6, (18.4, 25.3, 28.4, 29.3, 33, 33.3, 33.8, 34.4, 35.5, 37)),
    '32s': (91.0, (17.8, 24.9, 28.1, 30.4, 32.8, 32.4, 34.3, 33.7, 34.8, 37.1)),
    '73': (91.9, (14.3, 19.3, 26.1, 28.5, 27.6, 31, 31.6, 31.5, 31.9, 36.5)),
    '53': (92.8, (17.8, 22.8, 27.8, 28.6, 31.5, 32.9, 32.4, 33.7, 33.8, 37)),
    '82': (93.7, (12.5, 19.5, 25.6, 25.1, 28.3, 28.7, 29.8, 30.2, 31.2, 36.7)),
    '63': (94.6, (14.8, 21.1, 26.8, 28.3, 30.1, 31, 32.9, 31.8, 33.6, 36.8)),
    '43': (95.5, (16.6, 23, 28.1, 29.3, 29.2, 31.8, 33.2, 33.5, 32.3, 35.6)),
    '72': (96.4, (12, 17.6, 24.5, 25.2, 27.8, 28.1, 30.7, 30.3, 30, 34.4)),
    '62': (97.3, (12.3, 18.8, 25.1, 26.5, 27.9, 30.1, 30.5, 31.3, 31.1, 34.6)),
    '52': (98.2, (14.8, 21.3, 25.8, 27.9, 30.1, 30.8, 30.4, 31.5, 33.4, 34.3)),
    '42': (99.1, (14, 21.2, 26.3, 27.5, 28.5, 30.1, 31.3, 30.7, 30.8, 33.8)),
    '32': (100.0, (14.2, 20.5, 26, 27.2, 29.1, 28.9, 30.4, 32.6, 31.4, 31)),
}

# Opponent tracking (the bot's process stays alive between calls)
_seen = {}      # id -> preflop hands observed
_shoves = {}    # id -> preflop all-ins observed
_last_hand = None
_shoved_this_hand = set()


def chen(cards) -> float:
    a, b = sorted(cards, key=lambda c: VALUE[c.rank], reverse=True)
    hi, lo = VALUE[a.rank], VALUE[b.rank]
    pts = lambda v: CHEN_POINTS.get(v, v / 2)
    if hi == lo:
        return 6 if hi == 5 else max(pts(hi) * 2, 5)
    gap = hi - lo - 1
    score = pts(hi) - GAP_PENALTY.get(gap, 5)
    if a.suit == b.suit:
        score += 2
    if gap <= 1 and hi < 12:
        score += 1
    return score


def straight_high(values):
    s = set(values)
    if 14 in s:
        s.add(1)
    for top in range(14, 4, -1):
        if all(v in s for v in range(top - 4, top + 1)):
            return top
    return 0


def classify(hole, board):
    """Returns (strength, draw). strength: 0 nothing, 1 weak pair, 2 good pair, 3 two pair/set, 4 straight+."""
    cards = list(hole) + list(board)
    hv = [VALUE[c.rank] for c in hole]
    bv = [VALUE[c.rank] for c in board]
    counts = {}
    for c in cards:
        counts[VALUE[c.rank]] = counts.get(VALUE[c.rank], 0) + 1
    suits = {}
    for c in cards:
        suits[c.suit] = suits.get(c.suit, 0) + 1
    board_counts = {}
    for v in bv:
        board_counts[v] = board_counts.get(v, 0) + 1

    flush_suit = next((s for s, n in suits.items() if n >= 5), None)
    if flush_suit is not None and any(c.suit == flush_suit for c in hole):
        return 4, False
    st = straight_high(list(counts))
    if st and st > straight_high(bv) and len(bv) >= 3:
        return 4, False

    # Pairs/trips/quads that use at least one hole card
    multi = sorted(((n, v) for v, n in counts.items() if n >= 2 and n > board_counts.get(v, 0)), reverse=True)
    trips = [v for n, v in multi if n >= 3]
    pairs = [v for n, v in multi if n == 2]
    if trips or len(pairs) >= 2 or (pairs and max(board_counts.values(), default=0) >= 2):
        strength = 3
    elif pairs:
        p = pairs[0]
        top_board = max(bv) if bv else 0
        strength = 2 if p >= top_board else 1
        if strength == 2 and p == top_board and min(hv) < 9 and max(hv) == p and p < 11:
            strength = 2  # top pair, weak kicker is still fine at this level
    else:
        strength = 0

    draw = False
    if len(bv) < 5:
        if any(n == 4 and any(c.suit == s for c in hole) for s, n in suits.items()):
            draw = True
        vals = set(counts)
        if 14 in vals:
            vals.add(1)
        for low in range(1, 11):
            if sum(v in vals for v in range(low, low + 5)) == 4 and any(v in range(low, low + 5) for v in hv):
                draw = True
                break
    return strength, draw


def safe_call(to_call):
    return Action.check() if to_call <= 0 else Action.call()


def bet(state, me, to_call, amount, can_raise):
    if not can_raise:
        return safe_call(to_call)
    amount = max(int(amount), state.min_raise, 1)
    return Action.raise_by(amount)


def shove(state, me, to_call, can_raise):
    return bet(state, me, to_call, me.chips_in_stack + 1, can_raise)


def track(state):
    global _last_hand, _shoved_this_hand
    me = state.my_player
    key = frozenset(me.hole_cards)
    if state.community_cards:
        return
    if key != _last_hand:
        _last_hand = key
        _shoved_this_hand = set()
        for p in state.players:
            if p.id is not None and p.id != me.id:
                _seen[p.id] = _seen.get(p.id, 0) + 1
    for p in state.players:
        if p.id is None or p.id == me.id or p.id in _shoved_this_hand:
            continue
        if not p.has_folded and p.chips_in_stack == 0 and p.chips_bet > 0 and p.chips_bet > 2 * state.big_blind:
            _shoved_this_hand.add(p.id)
            _shoves[p.id] = _shoves.get(p.id, 0) + 1


LETTERS = dict(zip(RANKS, "23456789TJQKA"))


def hand_key(cards):
    a, b = sorted(cards, key=lambda c: VALUE[c.rank], reverse=True)
    if a.rank == b.rank:
        return LETTERS[a.rank] * 2
    return LETTERS[a.rank] + LETTERS[b.rank] + ("s" if a.suit == b.suit else "")


def equity_vs(key, range_pct):
    """Our equity (0-1) against an opponent who holds the top range_pct% of hands."""
    eqs = HANDS[key][1]
    r = max(RANGE_PCTS[0], min(100, range_pct))
    for i in range(len(RANGE_PCTS) - 1):
        lo, hi = RANGE_PCTS[i], RANGE_PCTS[i + 1]
        if r <= hi:
            t = (r - lo) / (hi - lo)
            return (eqs[i] + t * (eqs[i + 1] - eqs[i])) / 100
    return eqs[-1] / 100


def shove_range(p, bb):
    """Estimated % of hands this opponent goes all in (or bets big) with."""
    seen = _seen.get(p.id, 0)
    rate = (_shoves.get(p.id, 0) + 0.1 * 4) / (seen + 4)  # prior: shoves about 10% of hands
    pct = 100 * rate
    if (p.chips_bet + p.chips_in_stack) / max(bb, 1) < 10:
        pct = max(pct, 50)  # short stacks shove much wider (the example bots shove any two cards)
    return max(4, min(100, pct * 0.8))


def call_allin_ok(state, me, to_call, key, raisers, bb_left):
    """Equity against the shover's estimated range, compared with the pot odds."""
    bb = state.big_blind
    pot_after = state.chips_in_pot + min(to_call, me.chips_in_stack)
    need = min(to_call, me.chips_in_stack) / max(pot_after, 1)
    tightest = min((shove_range(p, bb) for p in raisers), default=10)
    eq = equity_vs(key, tightest)
    for _ in raisers[1:]:
        eq *= 0.7
    margin = -0.08 if bb_left >= 20 else -0.05  # tuned: demand a little more equity than the pot odds
    return eq >= need - margin, eq


def preflop(state, me, to_call, can_raise, bb_left, big, key, live):
    bb = state.big_blind
    chen_score = chen(me.hole_cards)
    max_bet = max(p.chips_bet for p in state.players)
    raisers = [p for p in live if p is not me and p.chips_bet > bb and p.chips_bet >= max_bet * 0.5]
    unopened = max_bet <= bb

    # Short stack: push or fold
    if bb_left <= 13:
        thr = 9 if bb_left > 10 else 7 if bb_left > 7 else 5 if bb_left > 4 else -10
        if unopened:
            return shove(state, me, to_call, can_raise) if chen_score >= thr else safe_call(to_call) if to_call <= 0 else Action.fold()
        ok, eq = call_allin_ok(state, me, to_call, key, raisers, bb_left)
        return shove(state, me, to_call, can_raise) if ok else Action.fold()

    if unopened:
        if chen_score >= 12 and bb_left <= 25:
            return shove(state, me, to_call, can_raise)
        if chen_score >= 2 or (chen_score >= 5 and len(live) <= 3):
            return bet(state, me, to_call, bb * 1 + (len(state.players) - len(live)) // 3 * bb, can_raise)
        return safe_call(to_call) if to_call <= 0 else Action.fold()

    # Facing a raise
    if big:  # all in or a big chunk of our stack
        ok, eq = call_allin_ok(state, me, to_call, key, raisers, bb_left)
        if not ok:
            return Action.fold()
        return shove(state, me, to_call, can_raise) if eq > 0.6 else Action.call()
    if chen_score >= 12:
        return shove(state, me, to_call, can_raise) if bb_left <= 35 else bet(state, me, to_call, to_call * 3, can_raise)
    if chen_score >= 9 or (chen_score >= 7 and to_call <= 3 * bb):
        return Action.call()
    return Action.fold()


def postflop(state, me, to_call, can_raise, live):
    strength, draw = classify(me.hole_cards, state.community_cards)
    pot = max(state.chips_in_pot, 1)
    stack = me.chips_in_stack
    river = len(state.community_cards) >= 5
    heads_up = len(live) <= 2
    committed = stack <= pot * 0.6

    if strength >= 3:
        if committed:
            return shove(state, me, to_call, can_raise)
        return bet(state, me, to_call, pot if to_call == 0 else pot + to_call, can_raise)

    if strength == 2:
        if to_call == 0:
            return bet(state, me, to_call, pot * 0.66, can_raise)
        if committed or to_call <= pot * 0.9 or to_call <= stack * 0.35:
            return Action.call()
        return Action.fold()

    if draw and not river:
        if to_call == 0:
            if committed:
                return shove(state, me, to_call, can_raise)
            return bet(state, me, to_call, pot * 0.5, can_raise) if random.random() < 0.5 else Action.check()
        if to_call <= pot * 0.45 or committed:
            return Action.call()
        return Action.fold()

    if strength == 1:
        if to_call == 0:
            return Action.check()
        return Action.call() if to_call <= pot * 0.3 else Action.fold()

    if to_call == 0:
        if heads_up and random.random() < 0.3:
            return bet(state, me, to_call, pot * 0.5, can_raise)
        return Action.check()
    return Action.fold()


def decide(state: GameState) -> Action:
    me = state.my_player
    max_bet = max((p.chips_bet for p in state.players), default=0)
    to_call = max(0, max_bet - me.chips_bet)
    if me.chips_in_stack <= 0:
        return Action.check() if to_call == 0 else Action.call()
    if len(me.hole_cards) != 2:
        return safe_call(to_call) if to_call == 0 else Action.fold()

    track(state)
    bb = max(state.big_blind, 1)
    live = [p for p in state.players if not p.has_folded]
    others_can_act = any(p is not me and not p.has_folded and p.chips_in_stack > 0 for p in state.players)
    can_raise = others_can_act and me.chips_in_stack > to_call
    bb_left = (me.chips_in_stack + me.chips_bet) / bb
    big = to_call >= 0.35 * (me.chips_in_stack + me.chips_bet) or any(
        p is not me and not p.has_folded and p.chips_in_stack == 0 and p.chips_bet == max_bet and max_bet > bb for p in state.players)

    if not state.community_cards:
        return preflop(state, me, to_call, can_raise, bb_left, big, hand_key(me.hole_cards), live)
    return postflop(state, me, to_call, can_raise, live)


def myAI(state: GameState) -> Action:
    try:
        action = decide(state)
        if isinstance(action, Action):
            return action
    except Exception:
        pass
    return defaultAI(state)


def defaultAI(state: GameState) -> Action: # This ai runs if myAI raises an exception, returns an invalid action or takes too long
    return Action.check() if not state.my_player.chips_in_stack or state.my_player.chips_bet == max(player.chips_bet for player in state.players) else Action.fold()