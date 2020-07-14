function Afv_x = deriv_exp_func(fx)
        Afv_x = fx .* (1 - fx);
end