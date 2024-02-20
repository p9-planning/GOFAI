;; cars=19, locations=15, out_folder=training/easy, instance_id=93, seed=123

(define (problem ferry-93)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 car8 car9 car10 car11 car12 car13 car14 car15 car16 car17 car18 car19 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 loc10 loc11 loc12 loc13 loc14 loc15 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc1)
    (at car1 loc5)
    (at car2 loc2)
    (at car3 loc13)
    (at car4 loc7)
    (at car5 loc5)
    (at car6 loc2)
    (at car7 loc14)
    (at car8 loc15)
    (at car9 loc14)
    (at car10 loc1)
    (at car11 loc7)
    (at car12 loc9)
    (at car13 loc9)
    (at car14 loc6)
    (at car15 loc6)
    (at car16 loc14)
    (at car17 loc1)
    (at car18 loc3)
    (at car19 loc3)
)
 (:goal  (and (at car1 loc7)
   (at car2 loc10)
   (at car3 loc6)
   (at car4 loc13)
   (at car5 loc4)
   (at car6 loc4)
   (at car7 loc1)
   (at car8 loc7)
   (at car9 loc13)
   (at car10 loc3)
   (at car11 loc11)
   (at car12 loc7)
   (at car13 loc2)
   (at car14 loc1)
   (at car15 loc7)
   (at car16 loc12)
   (at car17 loc9)
   (at car18 loc2)
   (at car19 loc1))))
