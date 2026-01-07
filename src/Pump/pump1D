import numpy as np
from scipy.optimize import fsolve

from src.fluid import Fluid
from src.Utility.calc_shaft import min_impeller_r, calc_shaft_radius
from src.Utility.calc_shaft import calc_hydraulic_efficiency

pi = np.pi
gravity = 9.805


class VelocityTriangle:
    # Meridyenel bir eğrideki bir nokta için hız üçgeni kurgular. Kanatlar için kurgulanmıştır. Giriş:0 Çıkış:1
    # merid_loc : Hız üçgenin meridyenel eğri üzerindeki normalize konumu
    def __init__(self, i_o, parent):
        self.i_o = i_o
        self.parent = parent
        self._beta_blade = None
        self.incidence = None
        self.radius = None
        self._alpha = None

    @property
    def c_m(self):
        if self.i_o == "inlet":
            return self.parent.parent.vol_flow / self.parent.parent.inlet_area
        else:
            return self.parent.parent.vol_flow / self.parent.parent.outlet_area

    @property
    def c_m_bl(self):
        return self.c_m * self.blockage

    @property
    def pitch(self):
        return self.radius * 2 * pi / self.parent.parent.blade_number

    @property
    def t(self):
        if self.i_o == "inlet":
            return self.parent.thickness_array[0]
        else:
            return self.parent.thickness_array[-1]

    @property
    def blockage(self):
        return (1 - self.parent.parent.blade_number * self.t / pi / self.radius / 2 / np.sin(self.beta_blade)) ** -1

    @property
    def beta(self):
        return np.arctan(self.c_m / self.w_u)

    @property
    def beta_bl(self):
        return np.arctan(self.c_m_bl / self.w_u)

    @property
    def alpha(self):
        return np.arctan(self.c_m / self.c_u) if self.i_o == "outlet" else self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def alpha_bl(self):
        return np.arctan(self.c_m_bl / self.c_u)

    @property
    def beta_blade(self):
        if self.i_o == "inlet":
            dummy_beta_blade = 10 * pi / 180
            for i in range(1000):
                blockage = (1 - self.parent.parent.blade_number * self.t / pi / self.radius / 2 / np.sin(
                    dummy_beta_blade)) ** -1
                beta_bl = np.arctan(self.c_m * blockage / self.w_u)
                beta_blade = beta_bl + self.incidence
                if abs(beta_blade - dummy_beta_blade) < 1e-4 or i == 999:
                    return beta_blade
                else:
                    dummy_beta_blade = beta_blade
        else:
            return self._beta_blade

    @beta_blade.setter
    def beta_blade(self, value):
        self._beta_blade = value

    @property
    def deviation(self):
        return self.beta_blade - self.beta

    @property
    def deviation_bl(self):
        return self.beta_blade - self.beta_bl

    @property
    def slip_factor(self):
        # Gülich Eq. 6.10 ile kayma faktörü hesaplar.
        epsilon_lim = np.exp(
            -8.16 * np.sin(self.beta_blade * pi / 180) / self.parent.parent.blade_number)
        d1m_star = self.parent.inlet.radius / self.radius
        k_w = 1 if d1m_star < epsilon_lim else 1 - ((d1m_star - epsilon_lim) / (1 - epsilon_lim)) ** 3
        f1 = 0.98
        return f1 * (1 - np.sqrt(
            np.sin(self.beta_blade)) / self.parent.parent.blade_number ** 0.7) * k_w if self.i_o == "outlet" else 0

    @property
    def w_u(self):
        return self.u - self.c_u

    @property
    def c_u(self):
        if self.i_o == "inlet":
            return self.c_m / np.tan(self.alpha)
        else:
            if self.parent.parent.__class__.__name__ == "Inducer":
                deviation = ((2 / 180 * pi + (self.beta_blade - self.parent.inlet.beta_blade) / 3) *
                             np.cbrt(1 / self.parent.parent.l_over_t))
                beta = self.beta_blade - deviation

                return self.u - self.c_m / np.tan(beta)
            else:
                return self.u * (self.slip_factor - self.c_m * self.blockage / self.u / np.tan(self.beta_blade))

    @property
    def flow_number(self):
        return self.c_m / self.u

    @property
    def flow_number_bl(self):
        return self.c_m_bl / self.u

    @property
    def w(self):
        return np.sqrt(self.w_u ** 2 + self.c_m ** 2)

    @property
    def w_bl(self):
        return np.sqrt(self.w_u ** 2 + self.c_m_bl ** 2)

    @property
    def c(self):
        return np.sqrt(self.c_u ** 2 + self.c_m ** 2)

    @property
    def c_bl(self):
        return np.sqrt(self.c_u ** 2 + self.c_m_bl ** 2)

    @property
    def u(self):
        # Dönme hızı ile hız üçgeninin bulunduğu konumdaki dönme çigisel hızını hesaplar.
        return self.radius * self.parent.parent.omega

    # def plot_triangle(self, ax=None):
    #     if ax is None:
    #         fig = Figure()
    #         ax = fig.add_subplot()
    #
    #     ax.plot([0, self.u], [0, 0])  # u
    #     ax.plot([0, self.w_u], [0, self.c_m])  # w
    #     ax.plot([self.u, self.w_u], [0, self.c_m])  # c
    #     ax.plot([0, self.w_u], [0, self.c_m_bl])  # w_bl
    #     ax.plot([self.u, self.w_u], [0, self.c_m_bl])  # c_bl


class MeridionalCurve:
    # Kanat üzerindeki herhangi bir spanda meridyenel eğriyi oluşturur.
    #   span: hub'dan tip'e 0...1
    #   parent: MeridionalCurve'ü çağıran obje, Rotor tabanlı bir sınıf olmalı.
    # TODO: Kontrol: parent, class oluşturulduktan sonra güncellenmiyor. Parent dışarıda değişirse burda güncellenecek mi?
    def __init__(self, parent, span):
        self.parent = parent
        self.parent: Rotor
        self.static_head = 0
        self.head = 0
        self.inlet = VelocityTriangle("inlet", parent=self)
        self.outlet = VelocityTriangle("outlet", parent=self)
        self.span = span
        self.thickness_matris = np.array([[0, 1], [2e-3, 2e-3]])
        self.deHaller_criteria = 0.7  # Güclih Heading 7.7.1.16, minimum [0.7]
        self.deHaller_ok = False

    @property
    def thickness_array(self):
        return np.interp(np.linspace(0, 1, 100), self.thickness_matris[0], self.thickness_matris[1])

    def check_deHaller(self):
        # Güclih Heading 7.7.1.16, minimum [0.7]
        # TODO: deHaller kriteri head hesabına eklenecek.
        self.deHaller_ok = False if self.outlet.w / self.inlet.w < self.deHaller_criteria else True

    def calc_head(self):
        # Önçark için Euler denklemi ile head, Gülich Eq. 7.6.15 ile static_head hesapla
        # Çark için Gülich Eq. 7.1.9 ile head, Gülich Eq. 3.3.8 ile static_head hesaplar.
        # TODO: static_head impeller kontrolü yapılacak.

        if self.parent.__class__.__name__ == "Inducer":
            self.head = (
                                self.outlet.c_u * self.outlet.u - self.inlet.c_u * self.inlet.u) / gravity * self.parent.hyd_eff
            self.static_head = (self.parent.hyd_eff * self.outlet.u ** 2 / 2 / gravity *
                                (1 - (self.parent.inlet_area ** 2 / self.parent.outlet_area ** 2 /
                                      np.sin(self.outlet.beta) ** 2 - 1) * self.inlet.flow_number ** 2))
        else:
            d1_star = self.inlet.radius / self.outlet.radius

            self.head = ((self.outlet.slip_factor - self.parent.vol_flow / self.parent.outlet_area / self.outlet.u *
                          (self.outlet.blockage + self.parent.outlet_area * d1_star *
                           np.tan(self.outlet.beta_blade) / self.parent.inlet_area / np.tan(self.inlet.alpha)) /
                          np.tan(
                              self.outlet.beta_blade)) * self.parent.hyd_eff * self.outlet.u ** 2 / gravity)
            self.static_head = ((self.outlet.u ** 2 - self.inlet.u ** 2 + self.inlet.w ** 2 - self.outlet.w ** 2) /
                                2 / gravity)

    def calc_inlet(self, value, method="alpha", incidence=None):
        # Farklı giriş bilgilerine göre giriş hız üçgeni hesaplar.
        #       method: "alpha" ,"c_u", "swirl_number"
        #       value: method seçiminde belirtilen parametre değeri
        #       c_m: Eksenel Mutlak Hız [m/s], girilmezse önceki değeri kullanır.
        #       incidence: Giriş Kaydırma Açısı [rad], girilmezse önceki değeri kullanır.
        self.inlet.incidence = self.inlet.incidence if incidence is None else incidence * pi / 180
        if method == "alpha":
            self.inlet.alpha = value * pi / 180
        elif method == "swirl_number":
            if value == 1:
                self.inlet.alpha = pi / 2
            else:
                self.inlet.alpha = np.arctan(self.inlet.flow_number / (1 - value))
        elif method == "c_u":
            if value == 0:
                self.inlet.alpha = pi / 2
            else:
                self.inlet.alpha = np.arctan(self.inlet.c_m / value)

    def calc_outlet_with_blockage(self, beta_blade=None):
        # Kanat açısı ile çıkış hız üçgeni hesaplar. Girdiler verilmezse güncelleme fonksiyonuna dönüşür.
        #       beta_blade: Kanat Açısı [rad]
        #       c_m: Eksenel Mutlak Hız [m/s]
        self.outlet.beta_blade = self.outlet.beta_blade if beta_blade is None else beta_blade

    def calc_beta_blade(self, head):
        # Euler denklemi ile numerik yöntem kullanarak verilen head için kanat açısı hesaplar.
        #       head: basma yüksekliği [m]
        def calc(beta_blade, h):
            beta_blade = beta_blade[0]
            deviation = ((2 / 180 * pi + (beta_blade - self.inlet.beta_blade) / 3) *
                         np.cbrt(1 / self.parent.l_over_t))
            beta = beta_blade - deviation
            w_u = self.outlet.c_m / np.tan(beta)
            c_u = self.outlet.u - w_u

            head_calc = (self.outlet.u * c_u) * self.parent.hyd_eff / gravity
            return head_calc - h

        self.outlet.beta_blade = fsolve(calc, np.array(self.outlet.beta_blade), args=head)[0]


class Rotor:
    # Dönen parçaları ortaklaması için kurgulanmıştır.
    def __init__(self, vol_flow, head_req, omega, shaft_radius, blade_number=6):
        #       vol_flow: Hacimsel Debi [m3/s],
        #       head_req: Hedef Basma Yüksekliği [m],
        #       omega: Dönme Hızı [rad/s],
        #       shaft_radius: Şaft Çapı [m],
        #       blade_number: Kanat Sayısı []
        self.vol_flow = vol_flow
        self.head_req = head_req
        self.omega = omega
        self.rpm = omega * 30 / np.pi
        self.shaft_radius = shaft_radius
        self.blade_number = blade_number

        self.head_rise = 0
        self.static_head_rise = 0
        self.npsh_required = 0
        self.suction_specific_speed = None  # Gülich Heading 6.2.4
        self.cavitation_number = None  # Gülich Heading 6.2.4
        self.hyd_eff = None
        self.vol_eff = 1

        self.width = None

        self.inlet_delta_c = None  # Gülich Heading 6.3.2, sadece NPSHr hesabı için
        self.inlet_delta_w = None  # Gülich Heading 6.3.2, sadece NPSHr hesabı için

        self.hub = MeridionalCurve(parent=self, span=0)
        self.tip = MeridionalCurve(parent=self, span=1)

        self.inlet_kn = None
        self.outlet_kn = None
        # self.inlet_area = 0
        self.outlet_area = 0

    @property
    def inlet_area(self):
        return pi * (self.tip.inlet.radius ** 2 - self.hub.inlet.radius ** 2)

    @property
    def specific_speed(self):  # [rpm,m3/s,m]
        return self.rpm * np.sqrt(self.vol_flow) / np.power(self.head_req, 0.75)

    @property
    def head_coeff_opt(self):
        return 1.21 * np.exp(-0.77 * self.specific_speed / 100)  # Gülich Eq. 3.26

    def calc_inlet(self, method="nss", c_u: list[float, float] = None, nss=1200):
        # Farklı metotlar ile rotor girişi hesaplanır.
        #       method: "nss",
        #           nss: Emme Özgül Hızı [m,m3/s,dev/dk]
        #       method: "nss",
        #           c_u: Mutlak Hız Teğetsel Bileşen Çifti [hub:[m/s], tip:[m/s]
        if method == "nss":
            self.suction_specific_speed = nss
            target_flow_number = 0.15 * (400 / nss) ** 0.93  # Gülich Eq 7.6.3
            k = self.vol_flow / target_flow_number / self.omega / pi
            for i in range(100):
                a = self.hub.inlet.radius ** 2
                b = np.cbrt(np.sqrt(3 * (27 * k ** 2 - 4 * a ** 3)) + 9 * k)
                self.tip.inlet.radius = b / np.cbrt(2) / np.power(3, 2 / 3) + np.cbrt(2 / 3) * a / b
                hub_ratio = self.hub.inlet.radius / self.tip.inlet.radius
                if hub_ratio < 0.15:  # Gülich Heading 7.7.1.5
                    self.hub.inlet.radius += 1e-5
                else:

                    self.hub.inlet.alpha = self.tip.inlet.alpha = np.pi / 2
                    blockage_coeff = 1.6  # Gülich Eq. 7.6.6, !!!! Sadece Uç Span İçin !!!!
                    design_incidence = 0.3
                    self.tip.inlet.incidence = np.arctan(self.tip.inlet.flow_number) * blockage_coeff * design_incidence
                    self.hub.inlet.incidence = np.arctan(self.hub.inlet.flow_number) * blockage_coeff * design_incidence
                    break

        if method == "minimum_relavite_velocity":
            if c_u is None:
                c_u = [15, 15]
            incidence = 0.0
            coeff = 1.15
            swirl_number_dummy = 1 - c_u[0] / self.hub.inlet.radius / self.omega / 1.5
            for i in range(100):
                # Minimum bağıl hız giriş hesaplar. İkincil akış için "coeff" ile çarpılır. Gülich Heading 7.2.1/6.A
                d_n_star = self.hub.inlet.radius / self.hub.outlet.radius
                self.tip.inlet.radius = self.hub.outlet.radius * coeff * np.sqrt(
                    d_n_star ** 2 + 1.48e-3 * self.head_coeff_opt * self.specific_speed ** 1.33 /
                    (swirl_number_dummy * self.vol_eff) ** 0.67)
                swirl_number = 1 - c_u[1] / self.tip.inlet.radius / self.omega
                if abs(swirl_number - swirl_number_dummy) <= 1e-5:
                    break
                else:
                    swirl_number_dummy = swirl_number

            self.hub.calc_inlet(incidence=incidence, value=c_u[0], method="c_u")
            self.tip.calc_inlet(incidence=incidence, value=c_u[1], method="c_u")

    def calc_suction_performance(self, delta_c=1.35):
        # Çarkın emme performansını hesaplar. Gülich Heading 6.3.2
        #       delta_c: giriş kayıp katsayısı
        self.inlet_delta_c = delta_c
        self.inlet_delta_w = 0.3 * np.tan(self.tip.inlet.beta) ** 0.57  # Gülich Fig.6.20
        c_m_max = np.max((self.hub.inlet.c_m_bl, self.tip.inlet.c_m_bl))
        w_max = np.max((self.hub.inlet.w_bl, self.tip.inlet.w_bl))
        alpha_max = np.max((self.hub.inlet.alpha_bl, self.tip.inlet.alpha_bl))
        hub_ratio = self.hub.inlet.radius / self.tip.inlet.radius
        k_n = 1 - hub_ratio ** 2
        self.npsh_required = ((self.inlet_delta_c * c_m_max ** 2 + self.inlet_delta_w * w_max ** 2)
                              / 2 / gravity)  # Gülich Eq. 6.10
        # self.npsh_required += self.head_loss_on_leading_edge
        self.cavitation_number = (
                (self.inlet_delta_c + self.inlet_delta_w) * self.tip.inlet.flow_number ** 2 +  # Gülich Eq. 6.11
                self.inlet_delta_w * (1 - self.tip.inlet.flow_number / np.tan(alpha_max)) ** 2)
        self.suction_specific_speed = (158 * np.sqrt(self.tip.inlet.flow_number * k_n) /  # Gülich Eq. 6.12
                                       self.cavitation_number ** 0.75)


class Impeller(Rotor):
    #  SI birimlerinde [m, s, Pa, rad], şoksuz giriş hesaplar, basma yüksekliğini çark çapı ile optimize eder.
    def __init__(self, vol_flow, head_req, omega, shaft_radius, blade_number=6):
        super().__init__(vol_flow=vol_flow, head_req=head_req, omega=omega, shaft_radius=shaft_radius,
                         blade_number=blade_number)
        #       vol_flow: Hacimsel Debi [m3/s],
        #       head_req: Hedef Basma Yüksekliği [m],
        #       omega: Dönme Hızı [rad/s],
        #       shaft_radius: Şaft Çapı [m],
        #       blade_number: Kanat Sayısı []
        self.e_star = 0.02
        self.head_coeff = None
        self.specific_diameter = None
        self.head_loss_on_leading_edge = 0  # hücum kenar profili kaynaklı kayıp

        self.vol_flow = vol_flow
        self.shaft_radius = shaft_radius
        self.blade_number = blade_number
        self.head_req = head_req

        self.vol_eff = 0.95
        self.hyd_eff = calc_hydraulic_efficiency(self.specific_speed)
        self.min_impeller_radius = min_impeller_r(shaft_radius)  # şaft yarıçapı + kama boşlşuğu + min et kalınlığı

        self.outlet_width = 0
        self.width = 0

        self.hub.inlet.radius = self.min_impeller_radius * 1.05
        self.calc_outlet_radius(self.head_coeff_opt)
        t = round(self.e_star * 84.6 / self.rpm * np.sqrt(self.head_req / self.head_coeff_opt), 4)
        self.hub.thickness_matris[1] = t
        self.tip.thickness_matris[1] = t
        self.set_outlet_beta_blade()
        self.calc_inlet(method="minimum_relavite_velocity")
        self.calc_outlet_width()
        self.calc_head()
        self.optimize_head()
        self.calc_width()
        self.calc_suction_performance()
        self.calc_specific_diameter()

    def calc_width(self):
        # Gülich Eq. 7.2 ile çark yüksekliği hesabı
        self.width = (2 * (self.tip.outlet.radius - self.tip.inlet.radius) * (self.specific_speed / 74) ** 1.07 +
                      self.outlet_width)

    def calc_outlet_radius(self, head_coeff, hub_or_tip="both"):
        # Gülich Eq 7.1.3 ile çark çıkış yarıçapı hesaplar.
        #       head_coeff: Head Coefficient
        #       hub_or_tip: Sadece göbek için "hub", sadece uç için "tip", her ikisi için "both"
        if hub_or_tip != "tip":
            self.hub.outlet.radius = 0.5 * 84.6 / self.rpm * np.sqrt(self.head_req / head_coeff)
        if hub_or_tip != "hub":
            self.tip.outlet.radius = 0.5 * 84.6 / self.rpm * np.sqrt(self.head_req / head_coeff)

    def calc_outlet_width(self, r2=None):
        # Gülich Eq. 7.1 ile çark çıkış yüksekliği hesaplar. Hesabı hub üzerinden yapar.
        #       r2: Çark Çıkış Yarıçapı [m]
        n_q_ratio = self.specific_speed / 100
        b2_star = 0.017 + 0.262 * n_q_ratio - 0.08 * n_q_ratio ** 2 + 0.0093 * n_q_ratio ** 3  # Gülich Eq. 7.1
        self.outlet_width = 2 * (r2 if r2 is not None else self.hub.outlet.radius) * b2_star
        self.outlet_area = pi * self.hub.outlet.radius * 2 * self.outlet_width
        c_m = self.vol_flow / self.outlet_area
        self.hub.calc_outlet_with_blockage()
        self.tip.calc_outlet_with_blockage()

    def calc_head(self):
        # Mevcut geometri için basma yüksekliği hesabı yapar.

        # r1m = (self.hub.inlet.radius + self.tip.inlet.radius) * 0.5
        # d1m_star = r1m / self.hub.outlet.radius
        # alpha1m = (self.hub.inlet.alpha + self.tip.inlet.alpha) * 0.5
        # c2_mean = (self.hub.outlet.c + self.tip.outlet.c) * 0.5
        # self.head_rise = (self.hub.outlet.slip_factor -  # Gülich Eq. 7.1.9
        #                   self.vol_flow / self.outlet_area / self.hub.outlet.u / np.tan(self.hub.outlet.beta_blade) *
        #                   (self.hub.outlet.blockage + self.outlet_area * d1m_star * np.tan(self.hub.outlet.beta_blade) /
        #                    self.inlet_area / np.tan(alpha1m))) * self.hyd_eff * self.hub.outlet.u ** 2 / gravity
        # self.static_head_rise = self.head_rise - c2_mean ** 2 / 2 / gravity
        self.hub.calc_head()
        self.tip.calc_head()
        self.head_rise = 0.5 * (self.hub.head + self.tip.head)
        self.head_coeff = 2 * gravity * self.head_rise / self.hub.outlet.u ** 2

    def optimize_head(self):
        # Gerekli basma yüksekliğine ulaşmak için çıkış yarıçapı hesaplar.
        err = self.head_rise - self.head_req
        for i in range(50):
            if err < 0:
                self.hub.outlet.radius *= 1.01
                self.tip.outlet.radius *= 1.01
            elif err > 10:
                self.hub.outlet.radius *= 0.99
                self.tip.outlet.radius *= 0.99
            else:
                break
            # self.calc_outlet_radius(self.head_coeff)
            self.calc_outlet_width()
            self.calc_head()
            err = self.head_rise - self.head_req

    def set_outlet_beta_blade(self, beta_blade=23.5 * pi / 180, hub_or_tip="both"):
        #  Kanat açısı ile çıkış hesaplar.
        #  Varsayılan ayarda Gülich 7.2.1.10'deki önerilen kanat açısı kullanılır.
        #       beta_blade: Kanat çıkış açısı [rad]
        #       hub_or_tip: Sadece göbek için "hub", sadece uç için "tip", her ikisi için "both"
        if hub_or_tip != "tip":
            self.hub.outlet.beta_blade = beta_blade
            self.hub.calc_outlet_with_blockage(beta_blade=beta_blade)
        if hub_or_tip != "hub":
            self.tip.outlet.beta_blade = beta_blade
            self.tip.calc_outlet_with_blockage(beta_blade=beta_blade)

    def calc_specific_diameter(self):
        self.specific_diameter = max(self.tip.outlet.radius, self.hub.outlet.radius) * \
                                 self.head_rise ** 0.25 / self.vol_flow ** 0.5

    def calc_brumfield_inlet(self, alpha: [float, float], incidence=1.0, brumfield_criteria=0.1):
        # TODO: Rotor sınıfına eklenecek.
        self.hub.inlet.radius = self.min_impeller_radius * 1.05
        brumfield_flow_number = np.sqrt(brumfield_criteria / (2 - 2 * brumfield_criteria))
        # debi sayısından analitik uç radius hesabı
        k = self.vol_flow / brumfield_flow_number / self.omega / pi
        a = self.hub.inlet.radius ** 2
        b = np.cbrt(np.sqrt(3 * (27 * k ** 2 - 4 * a ** 3)) + 9 * k)
        self.tip.inlet.radius = b / np.cbrt(2) / np.power(3, 2 / 3) + np.cbrt(2 / 3) * a / b
        self.hub.calc_inlet(incidence=incidence, value=alpha[0], method="alpha")
        self.tip.calc_inlet(incidence=incidence, value=alpha[1], method="alpha")


class Inducer(Rotor):
    # SI birimlerinde [m, s, Pa, rad], Nss ile giriş hesaplar, basma yüksekliğini kanat açısı ile optimize eder.
    # TODO: de Haller Criteria ve Lift Coefficient göre hız üçgeni hesabı eklenecek.
    # TODO: Brumfiled Coefficient ile giriş tasarımı eklenecek.
    def __init__(self, vol_flow, head_req, omega, shaft_radius, blade_number=3):
        super().__init__(vol_flow=vol_flow, head_req=head_req, omega=omega, shaft_radius=shaft_radius,
                         blade_number=blade_number)
        #       vol_flow: Hacimsel Debi [m3/s],
        #       head_req: Hedef Basma Yüksekliği [m],
        #       omega: Dönme Hızı [rad/s],
        #       shaft_radius: Şaft Çapı [m],
        #       blade_number: Kanat Sayısı []

        self.l_over_t = None  # Kanat uzunluğu / kanat adımı

        self.static_head_coeff = 0

        self.min_impeller_radius = self.shaft_radius + 3e-3  # Minimum et kalınlığı
        self.hub.inlet.radius = self.min_impeller_radius

        self.hub.thickness_matris[1] = [2e-4, 2e-3]
        self.tip.thickness_matris[1] = [1.5e-4, 1.5e-3]

        self.set_upstream()
        self.calc_inlet(method="nss")
        self.set_blade_length()
        self.set_outlet_radius(r_2_t=self.tip.inlet.radius, r_2_h=min_impeller_r(self.shaft_radius))
        self.set_outlet_blade_angle()
        self.calc_head()
        self.calc_outlet_blade_angle()
        self.calc_width()
        self.calc_suction_performance(delta_c=1.1)

    def calc_width(self):
        self.width = self.hub.inlet.pitch * self.l_over_t * 0.5 * (
                self.hub.inlet.beta_blade + self.hub.outlet.beta_blade)

    def set_upstream(self, alpha=90 * pi / 180):
        # Pompa giriş açısı tanımlar.
        self.hub.inlet.alpha = alpha
        self.tip.inlet.alpha = alpha

    def set_blade_length(self, l_over_t=2):
        # Gülich Eq. 7.6.12 ile kanat uzunluğu ve kanat adımı oranı ataması yapar.
        #       l_over_t: [1, 2.5]
        self.l_over_t = l_over_t
        self.calc_hyd_eff()

    def calc_hyd_eff(self):
        # Gülich Eq. 7.6.13 ile hidrolik verim hesaplanır.
        self.hyd_eff = 1 - 0.11 * self.l_over_t

    def set_outlet_radius(self, r_2_t=None, r_2_h=None):
        # Çıkışta göbek veya uç yarıçapı ataması yapar. Hız üçgenlerinin tekrar hesaplanması gerekir.
        #       r_2_t: Çıkış Uç Yarıçapı [m],
        #       r_2_h: Çıkış Göbek Yarıçapı [m],
        self.tip.outlet.radius = self.tip.outlet.radius if r_2_t is None else r_2_t
        self.hub.outlet.radius = self.hub.outlet.radius if r_2_h is None else r_2_h
        self.outlet_kn = 1 - (self.hub.outlet.radius / self.tip.outlet.radius) ** 2
        self.outlet_area = pi * self.tip.outlet.radius ** 2 * self.outlet_kn

    def set_outlet_blade_angle(self, beta_blade=15 * pi / 180, hub_or_tip="both"):
        #  Verilen kanat çıkış açısı ile göbek ve/veya uç hız üçgeni hesaplar.
        #       beta_blade: Kanat Çıkış Açısı
        #       hub_or_tip: Sadece göbek için "hub", sadece uç için "tip", her ikisi için "both"
        if hub_or_tip != "tip":
            self.hub.outlet.beta_blade = beta_blade
        if hub_or_tip != "hub":
            self.tip.outlet.beta_blade = beta_blade

    def calc_outlet_blade_angle(self):
        # Gerekli basınç artışını sağlayan kanat çıkış açısı göbek ve uç için hesaplanır.
        # TODO:
        #  Head kontrolü yapılacak.
        #  Alanı değişikliği entegre edilecek.
        self.calc_head()
        self.tip.calc_beta_blade(self.head_req)
        self.hub.calc_beta_blade(self.head_req)
        self.calc_head()

    def calc_head(self):
        # Mevcut Hız üçgenleri ile basma yüksekliği hesabı.
        # Gülich Eq. 7.6.15
        self.hub.calc_head()
        self.tip.calc_head()
        self.head_rise = (self.hub.head + self.tip.head) / 2
        self.static_head_rise = (self.hub.static_head + self.tip.static_head) / 2


class Volute:
    #  SI birimlerinde [m, s, Pa, rad], Impeller gereklidir.
    #  30 derecelik konik simetrik kesit ile salyangoz kayıpları hesaplar.
    #  TODO: Kesit tipleri eklenecektir.
    def __init__(self, fluid: Fluid, impeller: Impeller, vol_flow, outlet_radius, is_double_suction=False,
                 clearance=1e-3,
                 roughness=5e-6,
                 diffuser_expansion_angle=3.0, number_of_points=100):
        #       fluid: Akışkan, Fluid,
        #       impeller: Çark, Impeller
        #       vol_flow: Hacimsel Debi [m3/s],
        #       outlet_radius: Pompa Çıkış Yarıçapı [m],
        #       is_double_suction: Çark emiş adet kontrolu, True: Çift Emiş, False: Tek Emiş,
        #       clearance: Çark-Salyangoz arası açıklık [m],
        #       roughness: Salyangoz yüzey pürüzlülüğü [m]
        self.impeller = impeller
        self.fluid = fluid  # Kullanılan akışkan
        self.roughness = roughness  # Metal pürüzlülüğü
        self.vol_flow = vol_flow  # Volumetrik debi
        self.outlet_radius = outlet_radius
        self.diffuser_expansion_angle = diffuser_expansion_angle
        self.clearance = clearance  # Çark-Salyangoz açıklığı
        self.is_double_suction = is_double_suction

        self.cut_water = CutWater(self)
        self.cut_water.calc_radius()

        self.inlet_radius = 0
        self.inlet_height = 0
        self.inlet_area = 0
        self.inlet_cm = 0
        self.inlet_cu = 0
        self.inlet_c = 0
        self.inlet_alpha = 0  # Mutlak hız açısı (teğetsel ref.)

        self.set_inlet()

        self.angular_locs = np.linspace(self.cut_water.angular_loc, 360, 100)  # Hangi açısal kesitlerde hesap yapılsın.
        self.sections = np.zeros((len(self.angular_locs), number_of_points, 2))  # 2B kesit koordinatları
        self.section_areas = np.zeros_like(self.angular_locs)  # Kesit alanları
        self.section_wet_area = np.zeros_like(self.angular_locs)  # Kesit için dilim kalınlığındaki ıslak yüzey alanı
        self.section_lengths = np.zeros_like(self.angular_locs)
        self.section_velocity = np.zeros_like(self.angular_locs)  # Kesit ortalama hızı

        self.friction_head_loss = 0
        self.total_head_loss = 0
        self.shock_head_loss = 0

        self.calc_sections()

        self.exit_diff = ExitDiffuser(self, self.sections[-1], outlet_radius,
                                      diffuser_expansion_angle)

        self.calc_shock_head_loss()
        self.calc_friction_loss()
        self.calc_total_loss()

    def set_inlet(self):
        # Çark üstünde ve çark altında 1 mm boşluk varsayılmıştır. Çift emiş için disk kalınlığı 2 mm varsayılmıştır.
        self.inlet_radius = self.impeller.tip.outlet.radius + self.clearance
        self.inlet_height = self.impeller.outlet_width * 2 + 4e-3 if self.is_double_suction else self.impeller.outlet_width + 2e-3
        self.inlet_area = 2 * np.pi * self.inlet_radius * self.inlet_height
        outlet_c_m = 0.5 * (self.impeller.tip.outlet.c_m + self.impeller.tip.outlet.c_m)
        outlet_c_u = 0.5 * (self.impeller.tip.outlet.c_u + self.impeller.tip.outlet.c_u)
        self.inlet_cm = self.impeller.outlet_area * outlet_c_m / self.inlet_area
        self.inlet_cu = outlet_c_u * self.impeller.tip.outlet.radius / self.inlet_radius
        self.inlet_c = np.sqrt(self.inlet_cu ** 2 + self.inlet_cm ** 2)
        self.inlet_alpha = np.arctan(self.inlet_cm / self.inlet_cu)  # Mutlak hız açısı (teğetsel ref.)

    def calc_sections(self, cone_angle=30):
        # Konik kesitli salyangoz varsayımı yapılır. Koniklik açısı 30° varsayılır
        self.cone_angle = np.deg2rad(cone_angle)  # koniklik açısı radyan cinsinden depolanır.

        def calc_epsilon(c_l, e):
            # Scipy.fsolve için hazırlanmıştır.
            # Salyangozun hangi açısal konumunda, kesitin konik çizgi uzunluğunun ne kadar olacağını hesaplar.
            # Gülich Eq. 7.24' ü kullanır.
            # "height" ve "integral"in değiştirilmesi ile farklı kesit şekillerine uyarlanabilir.
            # Simetriklikten ötürü yarım kesit hesaplanır. Gerekli yerlerde iki ile çarpılır.
            r_start = 0.5 * (self.inlet_radius + self.cut_water.radius)
            r_end = self.inlet_radius + c_l * np.cos(self.cone_angle)
            radius, r_step = np.linspace(r_start, r_end, 100, retstep=True)
            section_slice_centers = (radius[1:] + r_step / 2).transpose()  # integral dilimlerinin merkezleri
            height = (section_slice_centers - r_start) * np.tan(self.cone_angle) + 0.5 * self.inlet_height
            self.section_velocity = np.sum(  # Alan ortalama
                self.inlet_cu * self.inlet_radius / section_slice_centers * (height.transpose() * r_step).transpose(),
                axis=1) / np.sum((height.transpose() * r_step).transpose(), axis=1)

            integral = np.tan(self.cone_angle) * (  # Kesite özgü analitik integral çözümü
                    (r_end - r_start) - r_start * np.log(r_end / r_start)) + 0.5 * self.inlet_height * np.log(
                r_end / r_start)
            epsilon_calc = self.inlet_cu * self.inlet_radius / self.vol_flow * 360 * 2 * integral
            return epsilon_calc - e

        # Salyangozun açısal konumu ile kesit büyüklüklerini eşleyen numerik çözüm.
        # Farklı kesit tipleri için kesit alanını kontrol eden bir parametre kurgulanmalı. Burada koninin çizgi uzunluğu
        # Başlangıç değeri, giriş yüksekliğinden türetilmiştir. (mertebe tutturmak yeterli)
        cone_lengths = fsolve(calc_epsilon,
                              np.linspace((self.cut_water.radius - self.inlet_radius) / np.cos(self.cone_angle),
                                          self.inlet_height,
                                          self.angular_locs.shape[0]),
                              args=self.angular_locs)

        # Kesitlerin r_max noktası koordinatları
        tip_points = np.array([self.inlet_radius + cone_lengths * np.cos(self.cone_angle),
                               self.inlet_height * 0.5 + cone_lengths * np.sin(self.cone_angle)])

        self.sections[:, :, 0] = np.linspace(self.inlet_radius, tip_points[0], 100).transpose()
        self.sections[:, :, 1] = ((self.sections[:, :, 0] - self.inlet_radius) *
                                  np.tan(self.cone_angle) + 0.5 * self.inlet_height)
        self.section_areas = ((self.sections[:, -1, 1] + 0.5 * self.inlet_height) *
                              (self.sections[:, -1, 0] - self.inlet_radius))
        self.section_lengths = 2 * (cone_lengths + self.sections[:, -1, 1])  # Kesitin çizgi uzunluğu

    def calc_friction_loss(self):
        #  Gülich Table 3.8(2), Eq. 3.8.21' göre hesaplanmış sürtünme katsayısı.
        #  Sadece salyangoz için hesaplanmaktadır.
        # TODO: Ortalama hız yerine her section için ayrı hız ve sürtünme katsayısı ile hesaplanacak.
        mean_profile_lengths = 0.5 * (self.section_lengths[1:] + self.section_lengths[:-1])
        profile_centers = self.sections[:, -1, 0] - (self.sections[:, -1, 0] - self.sections[:, 0, 0]) / 3
        angular_step = self.angular_locs[-1] - self.angular_locs[-2]
        arc_lengths = 0.5 * (profile_centers[1:] + profile_centers[:-1]) * angular_step * np.pi / 180
        self.section_wet_area = mean_profile_lengths * arc_lengths
        reynolds_volute = (0.5 * (self.section_velocity[1:] + self.section_velocity[:-1]) *
                           1.5 * arc_lengths.sum() / self.fluid.kinematicViscosity)
        c_f = 0.0015
        friction_coeff = 0.136 / (-np.log10(0.2 * self.roughness / 1.5 * arc_lengths.sum() +
                                            12.5 / reynolds_volute)) ** 2.15
        self.friction_head_loss = np.sum((0.5 * (self.section_velocity[1:] + self.section_velocity[:-1])) ** 3 *
                                         (friction_coeff + c_f) * self.section_wet_area) / self.vol_flow / gravity / 2

    def calc_shock_head_loss(self):
        # Salyangoz girişindeki şok kaybı hesabı.
        self.shock_head_loss = ((
                                        self.impeller.tip.outlet.blockage - self.impeller.outlet_width / self.inlet_height) ** 2 *
                                self.impeller.tip.outlet.flow_number ** 2) * self.impeller.tip.outlet.u ** 2 / gravity / 2
        if self.is_double_suction:
            self.shock_head_loss = ((
                                            self.impeller.tip.outlet.blockage - self.impeller.outlet_width * 2 / self.inlet_height) ** 2 *
                                    self.impeller.tip.outlet.flow_number ** 2) * self.impeller.tip.outlet.u ** 2 / gravity / 2

    def calc_total_loss(self):
        #  Toplam kaybı hesaplar.
        self.total_head_loss = self.friction_head_loss + self.shock_head_loss + self.exit_diff.head_loss

    def update_impeller(self, imp: Impeller):
        # Çarkta bir değişiklik olması durumunda salyangozu günceller.
        self.impeller = imp
        self.cut_water.calc_radius()
        self.set_inlet()
        self.calc_sections()
        self.exit_diff.update_inlet()
        self.calc_shock_head_loss()
        self.calc_friction_loss()
        self.calc_total_loss()


class ExitDiffuser:
    # Pompa çıkış difüzörü, performans hesabı ve genel boyutlandırma yapar.
    def __init__(self, parent: Volute, inlet_section: np.array, outlet_radius, cone_angle):
        #       parent: Salyangoz, Volute [m3/s],
        #       head_req: Hedef Basma Yüksekliği [m],
        #       omega: Dönme Hızı [rad/s],
        #       shaft_radius: Şaft Çapı [m],
        #       blade_number: Kanat Sayısı []
        self.parent = parent

        self.inlet_section = inlet_section
        self.inlet_radius = 0
        self.inlet_c = 0
        self.inlet_cm = 0

        self.static_head_rise = 0

        self.cone_angle = np.deg2rad(cone_angle)
        self.length = 0
        self.cp = 0  # basınç geri kazanım sayısı
        self.area_ratio = 1

        self.outlet_radius = outlet_radius
        self.outlet_cm = self.parent.vol_flow / self.outlet_radius ** 2 / np.pi
        self.outlet_c = 0

        self.head_loss = 0

        self.set_inlet()
        self.calc_diffuser()
        self.calc_loss()
        self.calc_outlet()

    def set_inlet(self):
        #  Çıkış difüzörü girişi hesaplanır.
        self.inlet_c = self.parent.section_velocity[-1]
        self.inlet_radius = (self.parent.section_areas[-1] / np.pi) ** 0.5
        self.inlet_cm = self.parent.vol_flow / self.inlet_radius ** 2 / np.pi

    def calc_diffuser(self):
        # Çıkış difüzörü hesaplanır. Gülich Heading 1.6
        self.area_ratio = self.outlet_radius ** 2 / self.inlet_radius ** 2
        self.length = (self.area_ratio - 1.05) * self.inlet_radius / 0.184
        self.cone_angle = np.arctan((self.outlet_radius - self.inlet_radius) / self.length)
        self.cp = 0.36 * (self.length / self.inlet_radius) ** 0.26
        self.static_head_rise = self.cp / 2 * self.inlet_c ** 2 / gravity

    def calc_loss(self):
        # Difüzör kaybı Gülich Table 3.8(2) Eq. 3.8.22
        self.head_loss = (
                self.inlet_c ** 2 * (1 - self.cp - 1 / self.area_ratio ** 2) * self.parent.vol_flow / 2 / gravity)

    def calc_outlet(self):
        # Difüzör çıkışı hesaplanır.
        self.outlet_c = ((
                                 self.inlet_c ** 2 / 2 / gravity - self.head_loss - self.static_head_rise) * 2 * gravity) ** 0.5

    def update_inlet(self):
        self.set_inlet()
        self.calc_diffuser()
        self.calc_loss()
        self.calc_outlet()


class CutWater:
    #  Sınıf düzenleme amacıyla kuruglanmıştır.
    def __init__(self, parent: Volute, thickness=2e-3, ang_loc=40):
        self.parent = parent
        self.radius = 0
        self.thickness = thickness
        self.angular_loc = np.deg2rad(ang_loc)  # Açısal yönde hangi derecede konumlanacak?

    def calc_radius(self):
        self.radius = ((1.03 + 0.1 * self.parent.impeller.specific_speed / 40 +
                        0.07 * self.parent.fluid.density / 1000 * self.parent.impeller.head_rise / 1000) *
                       self.parent.impeller.tip.outlet.radius)  # Gülich Table 10.2


class Diffuser:
    # TODO: Yeniden kurgulanacak.
    def __init__(self):
        # def __init__(self, Impeller, volumetric_flow, roughness):
        pass
        # # print("Diffuser is created")
        # self.volumetric_flow = volumetric_flow
        # self.inlet_3 = Location(Impeller.fluid, self.volumetric_flow, "radial")
        # self.outlet_4 = Location(Impeller.fluid, self.volumetric_flow, "radial")
        # self.inlet_5 = Location(Impeller.fluid, self.volumetric_flow, "radial")
        # self.outlet_6 = Location(Impeller.fluid, self.volumetric_flow, "radial")
        # inlet_3 = self.inlet_3
        # outlet_4 = self.outlet_4
        # inlet_5 = self.inlet_5
        # outlet_6 = self.outlet_6
        # self.blade = Blade(Impeller.fluid, Impeller.vol_flow, t1=0.01 * Impeller.outlet.radius, t2=0,
        #                    inlet_type="radial", outlet_type="radial")
        # inlet_3.width = Impeller.outlet.width * 1.1
        # self.incidence = 2 * pi / 180
        # self.number_of_blades = np.linspace(Impeller.blade_number, Impeller.blade_number + 4, 5)
        # if Impeller.specific_speed < 40:
        #     inlet_3.radius = Impeller.outlet.radius * (
        #             1.015 + 0.08 * ((Impeller.fluid.density * Impeller.head_rise_static) * 1e-6 - 0.1) ** 0.8)
        # else:
        #     inlet_3.radius = Impeller.outlet.radius * (1.04 + 0.001 * (Impeller.specific_speed - 40))
        #
        # inlet_3.c_u = Impeller.outlet.c_u * Impeller.outlet.radius / inlet_3.radius
        # inlet_3.blockage = Impeller.outlet.blockage
        # inlet_3.c_m = self.volumetric_flow / pi / inlet_3.radius / 2 / inlet_3.width
        # inlet_3.alpha = np.arctan(inlet_3.c_m / inlet_3.c_u) + self.incidence
        # x_le = np.exp(
        #     self.volumetric_flow / self.inlet_3.width / Impeller.outlet.c_u / Impeller.outlet.radius / self.number_of_blades)
        # inlet_3.width = 1.1 * inlet_3.radius * (x_le - 1)
        # inlet_3.radius_throat = inlet_3.radius + inlet_3.width * 0.5 * np.cos(inlet_3.alpha)
        # inlet_3.c_m_throat = inlet_3.c_m * inlet_3.radius / inlet_3.radius_throat
        # inlet_3.c_throat = 2 * pi * inlet_3.width * inlet_3.radius * inlet_3.c_m / inlet_3.width / self.inlet_3.width / self.number_of_blades
        # outlet_4.radius = Impeller.outlet.radius * (1.1 + 0.01 * Impeller.specific_speed)
        # self.lenght = np.dot(inlet_3.width.reshape((len(inlet_3.width), 1)),
        #                      np.arange(2.5, 6, 0.1).reshape((1, 35)))  # Gülich page 427 (5,35)
        # self.AR_opt = 1.05 + 0.184 * self.lenght / np.sqrt(
        #     inlet_3.width.reshape(len(inlet_3.width), 1) * inlet_3.width / pi)  # (5,35)
        # self.pressure_recovery_opt = 0.36 * (self.lenght / np.sqrt(
        #     inlet_3.width.reshape(len(inlet_3.width), 1) * inlet_3.width / pi)) ** 0.26  # (5,35)
        # self.delta_alpha = np.arange(0, 20, 2)
        # outlet_4.alpha = inlet_3.alpha + np.arange(0, 20, 2) * pi / 180
        # self.pressure_recovery_ideal = 1 - 1 / self.AR_opt ** 2  # (5,35)
        # self.pressure_recovery_loss = 0.5 * Impeller.fluid.density * (
        #         self.pressure_recovery_ideal - self.pressure_recovery_opt) * inlet_3.c_throat.reshape(
        #     len(inlet_3.c_throat), 1) ** 2
        # self.head_loss = self.pressure_recovery_loss / Impeller.fluid.density / gravity
        # inlet_3.c = np.sqrt(inlet_3.c_u ** 2 + inlet_3.c_m ** 2)
        # Re = inlet_3.c * self.lenght / Impeller.fluid.kinematicViscosity
        # Cf = 0.136 / (-np.log10(0.2 * roughness / self.lenght + 12.5 / Re)) ** 2.15
        #
        # a_star = inlet_3.width.reshape(len(inlet_3.width), 1) / 2 / Impeller.outlet.radius
        # b_star = inlet_3.width / 2 / Impeller.outlet.radius
        # b2_star = Impeller.outlet.width / 2 / Impeller.outlet.radius
        # zeta_semi_vaneless = (Cf + 0.0015) * (a_star + b_star) * pi ** 3 * (
        #         Impeller.outlet.flow_coeff * b2_star) ** 2 / (
        #                              8 * self.number_of_blades.reshape(len(self.number_of_blades),
        #                                                                1) * a_star * b_star) ** 3 * (
        #                              1 + Impeller.outlet.c / inlet_3.c_throat.reshape(len(inlet_3.c_throat),
        #                                                                               1)) ** 3
        #
        # self.head_loss_semi_vaneless = zeta_semi_vaneless * Impeller.outlet.u ** 2 / 2 / gravity
        # inlet_3.static_throat = (Impeller.fluid.density * gravity * (
        #         Impeller.outlet.head - self.head_loss_semi_vaneless) - 0.5 * Impeller.fluid.density * inlet_3.c_throat.reshape(
        #     len(inlet_3.c_throat), 1) ** 2)
        # outlet_4.static = self.pressure_recovery_opt * 0.5 * Impeller.fluid.density * inlet_3.c_throat.reshape(
        #     len(inlet_3.c_throat), 1) ** 2 + inlet_3.static
        #
        # inlet_3.area = inlet_3.width * inlet_3.width
        # outlet_4.width = inlet_3.width
        # outlet_4.area = self.AR_opt * inlet_3.area.reshape(len(inlet_3.area), 1)
        # outlet_4.width = outlet_4.area / outlet_4.width
        # outlet_4.radius_throat = outlet_4.radius - 0.5 * np.dot(
        #     outlet_4.width.reshape(outlet_4.width.shape[0], outlet_4.width.shape[1], 1),
        #     1 / np.cos(outlet_4.alpha).reshape(1, len(outlet_4.alpha)))
        # # print(outlet_4.radius_throat,outlet_4.radius_throat.shape)
        # outlet_4.c_m = self.volumetric_flow / 2 / pi / outlet_4.radius_throat / outlet_4.width
        # outlet_4.c = outlet_4.c_m / np.sin(outlet_4.alpha)
        # self.head_loss_return_channel = (1.5 * outlet_4.c ** 2 / Impeller.outlet.u ** 2)
        # self.head_loss_total = np.repeat((self.head_loss_semi_vaneless + self.head_loss)[:, :, np.newaxis],
        #                                  self.head_loss_return_channel.shape[2], axis=2) + self.head_loss_return_channel
        # ### DRAWING ###
        #
        # a, b, c = np.where(self.head_loss_total == self.head_loss_total.min())
        # self.number_of_blades = self.number_of_blades[a[0]]
        # self.AR_opt = self.AR_opt[a[0], b[0]]
        # self.head_loss = self.head_loss[a[0], b[0]]
        # self.head_loss_semi_vaneless = self.head_loss_semi_vaneless[a[0], b[0]]
        # self.lenght = self.lenght[a[0], b[0]]
        # self.pressure_recovery_ideal = self.pressure_recovery_ideal[a[0], b[0]]
        # self.pressure_recovery_opt = self.pressure_recovery_opt[a[0], b[0]]
        # self.pressure_recovery_loss = self.pressure_recovery_loss[a[0], b[0]]
        # self.delta_alpha = self.delta_alpha[c[0]]
        # self.head_loss_return_channel = self.head_loss_return_channel[a[0], b[0], c[0]]
        # self.head_loss_total = self.head_loss_total[a[0], b[0], c[0]]
        # self.total_pressure_loss = self.head_loss_total * Impeller.fluid.density * gravity
        # self.static_pressure_rise = self.pressure_recovery_opt * Impeller.fluid.density / 2 * self.inlet_3.c ** 2 - self.pressure_recovery_loss
        #
        # inlet_3.area = inlet_3.area[a[0]]
        # inlet_3.c_m_throat = inlet_3.c_m_throat[a[0]]
        # inlet_3.c_throat = inlet_3.c_throat[a[0]]
        # inlet_3.radius_throat = inlet_3.radius_throat[a[0]]
        # inlet_3.width = inlet_3.width[a[0]]
        # inlet_3.static_throat = inlet_3.static_throat[a[0], b[0]]
        # inlet_3.total = Impeller.outlet.total
        # inlet_3.head = inlet_3.total / Impeller.fluid.density / gravity
        # inlet_3.dynamic = inlet_3.c ** 2 * 0.5 * Impeller.fluid.density
        # inlet_3.static = inlet_3.total - inlet_3.static
        #
        # outlet_4.alpha = outlet_4.alpha[c[0]]
        # outlet_4.area = outlet_4.area[a[0], b[0]]
        # outlet_4.c = outlet_4.c[a[0], b[0], c[0]]
        # outlet_4.c_m = outlet_4.c_m[a[0], b[0], c[0]]
        # outlet_4.c_u = np.sqrt(outlet_4.c ** 2 - outlet_4.c_m ** 2)
        # outlet_4.dynamic = 0.5 * outlet_4.c ** 2 * Impeller.fluid.density
        # outlet_4.static = outlet_4.static[a[0], b[0]]
        # outlet_4.total = inlet_3.total - self.total_pressure_loss
        # outlet_4.head = outlet_4.total / Impeller.fluid.density / gravity
        # outlet_4.radius_throat = outlet_4.radius_throat[a[0], b[0], c[0]]
        # outlet_4.width = outlet_4.width[a[0], b[0]]
        #
        # # outlet_6.c_m = Impeller.inlet.mean.c_m * 0.85
        # # outlet_6.height = self.volumetric_flow / np.pi / Impeller.inlet.tip.radius / 2 / outlet_6.c_m
        # # outlet_6.alpha = 60 * np.pi / 180
        #
        # inlet_5.radius = outlet_4.radius
        # inlet_5.width = outlet_4.width
        # inlet_5.c_m = self.volumetric_flow / np.pi / inlet_5.radius / 2 / inlet_5.width
        # inlet_5.c_u = outlet_4.radius_throat * outlet_4.c_u / inlet_5.radius
        # inlet_5.alpha = np.arctan(inlet_5.c_m / inlet_5.c_u) + self.incidence
        # self.number_of_blades_return = np.ones(5) * 10
        # inlet_5.blockage = Impeller.outlet.blockage
        #
        # x_le_return = np.exp(
        #     self.volumetric_flow / self.inlet_5.width / inlet_5.c_u / inlet_5.radius / self.number_of_blades_return)
        #
        # inlet_5.width = 1.1 * inlet_3.radius * (x_le - 1)
        # inlet_5.radius_throat = inlet_5.radius + inlet_5.width * 0.5 * np.cos(inlet_5.alpha)
        # inlet_5.c_m_throat = inlet_5.c_m * inlet_5.radius / inlet_5.radius_throat
        # inlet_5.c_throat = 2 * pi * inlet_5.width * inlet_5.radius * inlet_5.c_m / inlet_5.width / inlet_5.width / self.number_of_blades
        # outlet_6.radius = Impeller.inlet.tip.radius
        # self.lenght_return = np.dot(inlet_5.width.reshape((len(inlet_5.width), 1)),
        #                             np.arange(2.5, 6, 0.1).reshape((1, 35)))  # Gülich page 427 (5,35)
        # self.AR_opt_return = 1.05 + 0.184 * self.lenght_return / np.sqrt(
        #     inlet_5.width.reshape(len(inlet_5.width), 1) * inlet_5.width / pi)  # (5,35)
        # self.pressure_recovery_opt_return = 0.36 * (self.lenght_return / np.sqrt(
        #     inlet_5.width.reshape(len(inlet_5.width), 1) * inlet_5.width / pi)) ** 0.26  # (5,35)
        # outlet_6.alpha = np.arange(50, 90, 2) * pi / 180
        # self.pressure_recovery_ideal_return = 1 - 1 / self.AR_opt_return ** 2  # (5,35)
        # self.pressure_recovery_loss_return = 0.5 * Impeller.fluid.density * (
        #         self.pressure_recovery_ideal_return - self.pressure_recovery_opt_return) * inlet_5.c_throat.reshape(
        #     len(inlet_5.c_throat), 1) ** 2
        # self.head_loss_return = self.pressure_recovery_loss_return / Impeller.fluid.density / gravity
        # inlet_5.c = np.sqrt(inlet_5.c_u ** 2 + inlet_5.c_m ** 2)
        #
        # inlet_5.static_throat = (Impeller.fluid.density * gravity * (
        #         outlet_6.head - self.head_loss_semi_vaneless) - 0.5 * Impeller.fluid.density * inlet_5.c_throat.reshape(
        #     len(inlet_5.c_throat), 1) ** 2)
        # outlet_6.static = self.pressure_recovery_opt_return * 0.5 * Impeller.fluid.density * inlet_5.c_throat.reshape(
        #     len(inlet_5.c_throat), 1) ** 2 + inlet_5.static
        #
        # inlet_5.area = inlet_5.width * inlet_5.width
        # outlet_6.width = inlet_5.width
        # outlet_6.area = self.AR_opt_return * inlet_5.area.reshape(len(inlet_5.area), 1)
        # outlet_6.width = outlet_6.area / outlet_6.width
        # outlet_6.radius_throat = outlet_6.radius - 0.5 * np.dot(
        #     outlet_6.width.reshape(outlet_6.width.shape[0], outlet_6.width.shape[1], 1),
        #     1 / np.cos(outlet_6.alpha).reshape(1, len(outlet_6.alpha)))
        # # print(outlet_4.radius_throat,outlet_4.radius_throat.shape)
        # outlet_6.c_m = self.volumetric_flow / 2 / pi / outlet_6.radius_throat / outlet_6.width
        # outlet_6.c = outlet_6.c_m / np.sin(outlet_6.alpha)
        #
        # self.head_loss_return_channel_return = (1.5 * outlet_6.c ** 2 / Impeller.outlet.u ** 2)
        # self.head_loss_total_return = np.repeat(
        #     (self.head_loss_semi_vaneless + self.head_loss_return)[:, :, np.newaxis],
        #     self.head_loss_return_channel_return.shape[2], axis=2) + self.head_loss_return_channel_return
        # ### DRAWING ###
        #
        # a, b, c = np.where(self.head_loss_total_return == self.head_loss_total_return.min())
        # self.AR_opt_return = self.AR_opt_return[a[0], b[0]]
        # self.head_loss_return = self.head_loss_return[a[0], b[0]]
        # # self.head_loss_semi_vaneless = self.head_loss_semi_vaneless[a[0], b[0]]
        # self.lenght_return = self.lenght_return[a[0], b[0]]
        # self.pressure_recovery_ideal_return = self.pressure_recovery_ideal_return[a[0], b[0]]
        # self.pressure_recovery_opt_return = self.pressure_recovery_opt_return[a[0], b[0]]
        # self.pressure_recovery_loss_return = self.pressure_recovery_loss_return[a[0], b[0]]
        # self.head_loss_return_channel_return = self.head_loss_return_channel_return[a[0], b[0], c[0]]
        # self.head_loss_total_return = self.head_loss_total_return[a[0], b[0], c[0]]
        # self.total_pressure_loss_return = self.head_loss_total_return * Impeller.fluid.density * gravity
        # self.static_pressure_rise_return = self.pressure_recovery_opt_return * Impeller.fluid.density / 2 * self.inlet_5.c ** 2 - self.pressure_recovery_loss_return
        #
        # inlet_5.area = inlet_5.area[a[0]]
        # inlet_5.c_m_throat = inlet_5.c_m_throat[a[0]]
        # inlet_5.c_throat = inlet_5.c_throat[a[0]]
        # inlet_5.radius_throat = inlet_5.radius_throat[a[0]]
        # inlet_5.width = inlet_5.width[a[0]]
        # # inlet_5.static_throat = inlet_5.static_throat[a[0], b[0]]
        # inlet_5.total = outlet_4.total
        # inlet_5.head = inlet_5.total / Impeller.fluid.density / gravity
        # inlet_5.dynamic = inlet_5.c ** 2 * 0.5 * Impeller.fluid.density
        # inlet_5.static = inlet_5.total - inlet_5.static
        #
        # outlet_6.alpha = outlet_6.alpha[c[0]]
        # outlet_6.area = outlet_6.area[a[0], b[0]]
        # outlet_6.c = outlet_6.c[a[0], b[0], c[0]]
        # outlet_6.c_m = outlet_6.c_m[a[0], b[0], c[0]]
        # outlet_6.c_u = np.sqrt(outlet_6.c ** 2 - outlet_6.c_m ** 2)
        # outlet_6.dynamic = outlet_6.c ** 2 * 0.5 * Impeller.fluid.density
        # outlet_6.static = outlet_6.static[a[0], b[0]]
        # outlet_6.total = inlet_5.total - self.total_pressure_loss_return
        # outlet_6.head = outlet_6.total / Impeller.fluid.density / gravity
        # outlet_6.radius_throat = outlet_6.radius_throat[a[0], b[0], c[0]]
        # outlet_6.width = outlet_6.width[a[0], b[0]]


class Station:
    #  Komponentler arası sınır koşulu bilgilerini tutar.
    #  Boş başlatılabilir
    #  Upstream veya downstream ve secondary_flows ile başlatılmalıdır. Verilmeyen hesaplanır.
    #  Özel durumlarda 3 girdi ile başlatılabilir.
    def __init__(self, upstream: list = None, downstream: list = None, secondary_flows: list = None):
        #       upstream: Yukarı-akış: list, [kg/s]
        #       downstream: Aşağı-akış: list, [kg/s]
        #       secondary_flows: İkincil akışlar: list, [kg/s]
        if secondary_flows is None:
            secondary_flows = [0]
        self.total_pressure = 0
        self.static_pressure = 0
        self.secondary_flows = np.array(secondary_flows)
        self.upstream_flow = np.array(upstream) if upstream is not None else np.array(
            [np.array(downstream).sum() - self.secondary_flows.sum()])
        self.downstream_flow = np.array(downstream) if downstream is not None else np.array(
            [np.array(upstream).sum() + self.secondary_flows.sum()])

    def update_secondary_flows(self, secondary_flows: list):
        self.secondary_flows = np.array(secondary_flows)
        self.upstream_flow = [self.downstream_flow.sum() - self.secondary_flows.sum()]


class Pump:
    # Pompa 1D hesaplamaları yapar. Genel boyutlandırma ve performans değerlerini çıktısı verir.
    # İstasyon kurgusu konfigürasyona göre sabittir. TODO: İstasyon kurgusu açıklanacak.
    __stressAllowable = 200e6  # [Pa]

    def __init__(self, rpm, pump_dict, fluid: Fluid, secondary_flows: dict = None):
        #       rpm: Dönme hızı, [dev/dk]
        #       pump_dict: Pompa sınır koşulları: dict
        #           {
        #               "mass_flow": kütlesel debi [kg/s],
        #               "pressure_required": lazım basınç [Pa],
        #               "inlet_pressure": giriş basıncı [Pa],
        #               "alpha": akışkan giriş açısı [deg],
        #               "shaft_radius": şaft çapı [m],
        #               "double_suction": çift emiş anahtarı [true/false],
        #               "second_stage": ikinci kademe anahtarı [true/false],
        #               "inlet_radius": giriş yarıçapı [m],
        #               "outlet_radius": çıkış yarıçapı [m]
        #           }
        #       fluid: akışkan, Fluid
        #       secondary_flows: ikincil akış kütlesel debileri: dict
        #             {
        #                 "station_3": list,
        #                 "station_2": list,
        #                 "station_2c": list,
        #                 "station_2b": list,
        #                 "station_2a": list,
        #                 "station_1": list,
        #                 "station_1s": list,
        #                 "station_1i": list,
        #                 "station_1is": list
        #             }
        self.input_dict = pump_dict
        self.rpm = rpm
        self.omega = rpm * np.pi / 30
        self.is_second_stage = pump_dict["second_stage"]
        self.is_double_suction = pump_dict["double_suction"]
        self.mass_flow = pump_dict["mass_flow"]
        self.inlet_angle = pump_dict["alpha"]
        self.fluid = fluid
        self.vol_flow = self.mass_flow / self.fluid.density
        __volumetric_flow_feet_s = self.vol_flow * 35.31467
        __volumetric_flow_gal_min = self.vol_flow * 15850.32
        __volumetric_flow_m_min = self.vol_flow * 60

        self.secondary_flows = {
            "station_3": [-0.025 * self.mass_flow],
            "station_2": [-0.025 * self.mass_flow, -0.025 * self.mass_flow],
            "station_2c": [0.025 * self.mass_flow],
            "station_2b": [0.025 * self.mass_flow],
            "station_2a": [0.025 * self.mass_flow],
            "station_1": [0.025 * self.mass_flow],
            "station_1s": [0.025 * self.mass_flow],
            "station_1i": [0.025 * self.mass_flow],
            "station_1is": [-0.025 * self.mass_flow]
        } if secondary_flows is None else secondary_flows

        self.station_3 = Station(downstream=[self.mass_flow],  # Pompa çıkış istasyonu
                                 secondary_flows=self.secondary_flows["station_3"])
        self.station_2 = Station(downstream=[self.station_3.upstream_flow[0]],  # Çark çıkış istasyonu
                                 secondary_flows=self.secondary_flows["station_2"])
        if self.is_second_stage:
            self.station_2c = Station(downstream=[self.station_3.upstream_flow[0]],  # 2. Çark çıkış istasyonu
                                      secondary_flows=self.secondary_flows["station_2c"])

            self.station_2b = Station(downstream=[self.station_2c.upstream_flow[0]],
                                      secondary_flows=self.secondary_flows["station_2b"])  # 2. Çark giriş istasyonu

            self.station_2a = Station(downstream=[self.station_2b.upstream_flow[0]],
                                      secondary_flows=self.secondary_flows["station_2a"])

            self.station_2.__init__(downstream=[self.station_2a.upstream_flow[0]],
                                    secondary_flows=self.secondary_flows["station_2"])  # Çark çıkış istasyonu

        if self.is_double_suction:
            self.station_2.__init__(upstream=np.concatenate((self.station_3.upstream_flow[0] / 2 +
                                                             self.secondary_flows["station_1"],
                                                             self.station_3.upstream_flow[0] / 2 +
                                                             self.secondary_flows["station_1s"])).tolist(),
                                    downstream=[self.station_3.upstream_flow[0]],
                                    secondary_flows=self.secondary_flows["station_2"])

        self.station_1 = Station(downstream=[self.station_2.upstream_flow[0]],  # Çark önü istasyonu
                                 secondary_flows=self.secondary_flows["station_1"])
        if self.is_double_suction:
            self.station_1s = Station(downstream=[self.station_2.upstream_flow[1]],  # Çark önü istasyonu
                                      secondary_flows=self.secondary_flows["station_1s"])

        self.station_0 = Station(downstream=[self.station_1.upstream_flow[0]])  # Pompa giriş istasyonu

        if self.is_double_suction:
            self.station_0.__init__(downstream=[self.station_1.upstream_flow[0],  # Pompa giriş istasyonu
                                                self.station_1s.upstream_flow[0]])

        self.inlet_radius = pump_dict["inlet_radius"]  # Pompa besleme hattı yarıçapı
        self.outlet_radius = pump_dict["outlet_radius"]  # Pompa tahliye hattı yarıçapı

        self.station_0.total_pressure = pump_dict["inlet_pressure"]
        self.station_0.static_pressure = (self.station_0.total_pressure -
                                          (self.station_0.upstream_flow.sum() / fluid.density /
                                           (pi * self.inlet_radius ** 2)) ** 2 / 2 * fluid.density)
        self.npsh_available = (self.station_0.total_pressure - fluid.vaporPressure) / self.fluid.density / gravity
        self.npsh_required = None
        self.pressure_req = pump_dict["pressure_required"]
        self.pressure_rise = self.pressure_req - self.station_0.total_pressure
        self.head_rise = self.pressure_rise / self.fluid.density / gravity
        self.head_req = self.pressure_req / self.fluid.density / gravity
        __head_rise_feet = self.head_rise * 3.28083

        self.specific_speed = self.rpm * np.sqrt(self.vol_flow) / np.power(self.head_rise, 0.75)
        specific_speed_rad = self.omega * np.sqrt(self.vol_flow) / np.power(gravity * self.head_rise, 0.75)
        self.specific_speed_rad_ft = self.rpm * np.sqrt(__volumetric_flow_feet_s) / np.power(__head_rise_feet, 0.75)
        specific_speed_rad_brennen = self.rpm * np.sqrt(__volumetric_flow_gal_min) / np.power(__head_rise_feet,
                                                                                              0.75) / 2734.6

        self.hydraulic_efficiency_initial = calc_hydraulic_efficiency(self.specific_speed)
        self.hydraulic_efficiency = None
        self.volumetric_efficiency = self.station_3.downstream_flow.sum() / self.station_0.upstream_flow.sum()
        self.mechanical_efficiency = 0.90
        self.total_efficiency = None

        self.hydraulic_work = self.vol_flow * self.pressure_rise
        self.head_loss_initial = self.hydraulic_work * (
                1 / self.hydraulic_efficiency_initial - 1) / gravity / self.mass_flow
        self.head_coeff_initial = 1.21 * np.e ** (
                -0.408 * self.specific_speed / 52.9)  # Gulich page 113 / equation 3.26
        self.shaft_power_initial = self.hydraulic_work / self.hydraulic_efficiency_initial / self.volumetric_efficiency / self.mechanical_efficiency

        if pump_dict["shaft_radius"]:
            self.shaft_radius = pump_dict["shaft_radius"]
        else:
            self.shaft_radius = calc_shaft_radius(h_w=self.hydraulic_work, s_a=self.__stressAllowable, rpm=self.rpm)

        self.head_loss = None
        self.head_coeff = None
        self.shaft_power = None

        self.inducer_need = False
        self.impeller = Impeller(vol_flow=self.station_1.upstream_flow[0] / self.fluid.density,
                                 head_req=self.head_rise * 1.05, omega=self.omega, shaft_radius=self.shaft_radius)
        self.update_station_for("impeller")

        self.volute = Volute(fluid=self.fluid, impeller=self.impeller,
                             vol_flow=self.station_2.downstream_flow[0] / self.fluid.density,
                             outlet_radius=self.outlet_radius, is_double_suction=self.is_double_suction)
        self.update_station_for("volute")

        self.check_inducer_need()
        if self.inducer_need:
            self.station_1i = Station(downstream=[self.station_1.upstream_flow[0]],
                                      secondary_flows=self.secondary_flows["station_1i"])

            self.station_0.__init__(downstream=[self.station_1i.upstream_flow[0]])  # Pompa giriş istasyonu

            if self.is_double_suction:
                self.station_1is = Station(downstream=[self.station_1s.upstream_flow[0]],
                                           secondary_flows=self.secondary_flows["station_1is"])
                self.station_0.__init__(downstream=[self.station_1i.upstream_flow[0],
                                                    self.station_1is.upstream_flow[0]])

            self.station_0.total_pressure = pump_dict["inlet_pressure"]
            self.station_0.static_pressure = (self.station_0.total_pressure -
                                              (self.station_0.upstream_flow.sum() / fluid.density /
                                               (pi * self.inlet_radius ** 2)) ** 2 / 2 * fluid.density)

            self.inducer = Inducer(vol_flow=self.station_1i.downstream_flow[0] / self.fluid.density,
                                   head_req=self.impeller.npsh_required, omega=self.omega,
                                   shaft_radius=self.shaft_radius)
            self.update_station_for("inducer")
            self.match_inducer_and_impeller()

        if self.is_second_stage:
            self.impeller2 = Impeller(vol_flow=self.station_2c.upstream_flow[0] / self.fluid.density,
                                      head_req=self.head_rise * 1.05, omega=self.omega, shaft_radius=self.shaft_radius)
            self.diffuser = Diffuser()
        self.volute.update_impeller(self.impeller)
        self.update_station_for("volute")
        self.optimize_impeller_head()
        self.calc_performance()

    def calc_performance(self):
        #  Pompa performans değerlerini hesaplar.
        self.shaft_power = 0
        self.head_rise = 0
        component_list = [self.impeller]
        self.npsh_required = self.impeller.npsh_required
        if self.inducer_need:
            component_list = [self.impeller, self.inducer]
            self.npsh_required = self.inducer.npsh_required
        for component in component_list:
            self.shaft_power += (component.head_rise * component.vol_flow * self.fluid.density * gravity /
                                 component.hyd_eff)
            self.head_rise += component.head_rise
        if self.is_double_suction:
            self.shaft_power *= 2
        self.hydraulic_efficiency = self.hydraulic_work / self.shaft_power
        self.total_efficiency = self.hydraulic_efficiency * self.mechanical_efficiency

    def optimize_impeller_head(self):
        # Pompa çıkış basıncını, lazım çıkış basıncına getirir.
        for i in range(100):
            if self.head_req > self.station_3.total_pressure / self.fluid.density / gravity:
                self.impeller.head_req = self.impeller.head_req * 1.1
            elif self.station_3.total_pressure / self.fluid.density / gravity - self.head_req > 10:
                self.impeller.head_req = self.impeller.head_req * 0.99
            else:
                break
            self.impeller.optimize_head()
            self.update_station_for("impeller")
            self.volute.update_impeller(self.impeller)
            self.update_station_for("volute")

    def update_station_for(self, comp: str):
        # Komponent işlemlerinden sonra istasyon bilgilerini günceller.
        if comp == "impeller":
            if not self.inducer_need:
                self.station_1.total_pressure = self.station_0.total_pressure - 1e5
                self.station_1.static_pressure = self.station_1.total_pressure - (
                        0.5 * (self.impeller.hub.inlet.c + self.impeller.tip.inlet.c)) ** 2 / 2 * self.fluid.density
            self.station_2.total_pressure = self.station_1.total_pressure + self.impeller.head_rise * self.fluid.density * gravity
            self.station_2.static_pressure = self.station_1.static_pressure + self.impeller.static_head_rise * self.fluid.density * gravity
        elif comp == "volute":
            self.station_3.total_pressure = self.station_2.total_pressure - self.volute.total_head_loss * self.fluid.density * gravity
            self.station_3.static_pressure = self.station_3.total_pressure - self.volute.exit_diff.outlet_c ** 2 / 2 * self.fluid.density
        elif comp == "inducer":
            self.station_1i.total_pressure = self.station_0.total_pressure - 1e5
            self.station_1i.static_pressure = self.station_1i.total_pressure - (
                    0.5 * (self.inducer.hub.inlet.c + self.inducer.tip.inlet.c)) ** 2 / 2 * self.fluid.density
            self.station_1.total_pressure = self.station_1i.total_pressure + self.inducer.head_rise * self.fluid.density * gravity
            self.station_1.static_pressure = self.station_1i.static_pressure + self.inducer.static_head_rise * self.fluid.density * gravity

    def check_inducer_need(self):
        # Önçark ihtiytacı kontrolü yapar.
        if self.impeller.npsh_required > self.npsh_available:
            self.inducer_need = True
        else:
            self.inducer_need = False

    def match_inducer_and_impeller(self):
        # Önçark çıkışı ile çark girişi günceller. Çark NPSH'ı ile önçark çıkışı hesaplar.
        for i in range(100):
            self.impeller.hub.inlet.radius = self.inducer.hub.outlet.radius
            self.impeller.tip.inlet.radius = self.inducer.tip.outlet.radius
            self.impeller.hub.calc_inlet(incidence=0, value=self.inducer.hub.outlet.c_u, method="c_u")
            self.impeller.tip.calc_inlet(incidence=0, value=self.inducer.tip.outlet.c_u, method="c_u")
            self.impeller.calc_suction_performance()
            head_req = self.impeller.npsh_required - self.station_1i.total_pressure / self.fluid.density / gravity
            self.inducer.head_req = head_req
            self.inducer.calc_outlet_blade_angle()
            self.inducer.calc_head()
            self.update_station_for("inducer")

            if self.impeller.npsh_required < self.station_1.total_pressure / self.fluid.density / gravity:
                break

        impeller_head_req = (self.head_req + self.volute.total_head_loss -
                             self.station_1.total_pressure / self.fluid.density / gravity)
        self.impeller.head_req = impeller_head_req
        self.impeller.optimize_head()
        self.update_station_for("impeller")
