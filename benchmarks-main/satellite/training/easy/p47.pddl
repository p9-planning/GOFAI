;; satellites=6, instruments=10, modes=2, directions=6, out_folder=training/easy, instance_id=47, seed=80

(define (problem satellite-47)
 (:domain satellite)
 (:objects 
    sat1 sat2 sat3 sat4 sat5 sat6 - satellite
    ins1 ins2 ins3 ins4 ins5 ins6 ins7 ins8 ins9 ins10 - instrument
    mod1 mod2 - mode
    dir1 dir2 dir3 dir4 dir5 dir6 - direction
    )
 (:init 
    (pointing sat1 dir3)
    (pointing sat2 dir4)
    (pointing sat3 dir5)
    (pointing sat4 dir6)
    (pointing sat5 dir4)
    (pointing sat6 dir3)
    (power_avail sat1)
    (power_avail sat2)
    (power_avail sat3)
    (power_avail sat4)
    (power_avail sat5)
    (power_avail sat6)
    (calibration_target ins1 dir3)
    (calibration_target ins2 dir5)
    (calibration_target ins3 dir4)
    (calibration_target ins4 dir3)
    (calibration_target ins5 dir1)
    (calibration_target ins6 dir5)
    (calibration_target ins7 dir4)
    (calibration_target ins8 dir2)
    (calibration_target ins9 dir5)
    (calibration_target ins10 dir1)
    (on_board ins1 sat6)
    (on_board ins2 sat1)
    (on_board ins3 sat3)
    (on_board ins4 sat5)
    (on_board ins5 sat2)
    (on_board ins6 sat4)
    (on_board ins7 sat2)
    (on_board ins8 sat4)
    (on_board ins9 sat5)
    (on_board ins10 sat2)
    (supports ins4 mod1)
    (supports ins7 mod2)
    (supports ins9 mod1)
    (supports ins2 mod1)
    (supports ins5 mod1)
    (supports ins10 mod1)
    (supports ins5 mod2)
    (supports ins1 mod1)
    (supports ins8 mod1)
    (supports ins8 mod2)
    (supports ins1 mod2)
    (supports ins7 mod1)
    (supports ins3 mod1)
    (supports ins2 mod2)
    (supports ins6 mod1))
 (:goal  (and (pointing sat1 dir3)
   (pointing sat3 dir2)
   (pointing sat6 dir4)
   (have_image dir2 mod1)
   (have_image dir6 mod2))))
