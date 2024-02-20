(define (problem satellite-07)
 (:domain satellite)
 (:objects 
    sat1 sat2 - satellite
    ins1 ins2 - instrument
    mod1 mod2 - mode
    dir1 dir2 dir3 - direction
    )
 (:init 
    (pointing sat1 dir3)
    (pointing sat2 dir3)
    (power_avail sat1)
    (power_avail sat2)
    (calibration_target ins1 dir1)
    (calibration_target ins2 dir2)
    (on_board ins1 sat2)
    (on_board ins2 sat1)
    (supports ins2 mod2)
    (supports ins2 mod1)
    (supports ins1 mod1))
 (:goal  (and
    (pointing sat2 dir3)
    (have_image dir1 mod1)
    (have_image dir2 mod2))))
