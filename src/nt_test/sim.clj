(ns nt-test.sim
  (:import java.util.concurrent.ThreadLocalRandom))

(defn dot [a b] (apply + (mapv * a b)))

(defn mv [m v]
  (let [f (fn [a & b] (apply + (map * a b)))]
    (apply mapv (partial f v) m)))

(defn tr [x]
  (apply map vector x))

(defn ones [v] (mapv (constantly 1) v))

(defn sum [v] (apply + v))

(defn mag [x] (Math/sqrt (dot x x)))
(defn mag-sq [x] (dot x x))
(defn dist [a b] (mag (mapv - a b)))
(defn dist-sq [a b] (mag-sq (mapv - a b)))

(defn logi [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn logi-adj [x] (/ 1.0 (+ 1.0 (Math/exp (* -4.9 x)))))

;; ThreadLocalRandom.current() is a static class method
;;  returns an rng with methods .nextInt etc.
(defn randbetween [a b]
  (let [r (ThreadLocalRandom/current)]
    (.nextInt r a b)))

(defn rand-double
  ([] (let [r (ThreadLocalRandom/current)]
        (.nextDouble r)))
  ([scale] (* scale (rand-double))))

(defn p-fn [pr a b]
  (fn [x] (if (< (rand-double) pr) (a x) (b x))))

(defn rand-gaussian
  ([] (let [r (ThreadLocalRandom/current)]
     (.nextGaussian r)))
  ([scale] (* scale (rand-gaussian))))

(defn rand-uniform
  ([] (let [r (ThreadLocalRandom/current)]
        (- (* 2.0 (.nextDouble r)) 1.0)))
  ([scale] (* scale (rand-uniform))))

(defn add-gaussian
  ([x] (+ (rand-gaussian) x))
  ([x scale] (+ (rand-gaussian scale) x)))

(defn add-uniform
  ([x] (+ (rand-uniform) x))
  ([x scale] (+ (rand-uniform scale) x)))

(defn rand-ints [a b]
  (repeatedly #(randbetween a b)))

(defn rand-int-vec [n a b]
  (vec (take n (rand-ints a b))))

(defn rand-floats []
  (repeatedly rand-double))

(defn rand-vec [n]
  (vec (take n (rand-floats))))

(defn pmap-vals [f m]
  (into {}
        (pmap
         (fn [[k v]] [k (f v)])
         m)))

(defn disj-vec [v & vals] (filterv (complement (set vals)) v))

(defn mergef [f & maps]
  (persistent!
   (reduce
    (fn [m-tr m2]
      (reduce (fn [m [k v]]
                (if-let [v0 (get m k)]
                  (assoc! m k (f v0 v))
                  (assoc! m k v))) m-tr m2))
    (transient (first maps))
    (rest maps))))

(def genkey (comp keyword gensym name))

