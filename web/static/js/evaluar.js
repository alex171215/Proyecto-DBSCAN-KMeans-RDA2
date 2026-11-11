// Formulario en 3 pasos: muestra un paso a la vez y valida los obligatorios antes de avanzar.
(function () {
  const form = document.querySelector("[data-form]");
  if (!form) return;
  const pasos = [...form.querySelectorAll("[data-paso]")];
  const indicador = [...document.querySelectorAll("[data-pasos] li")];
  let actual = 0;

  form.classList.add("con-js");

  function mostrar(i, desplazar = true) {
    actual = i;
    pasos.forEach((p, j) => p.classList.toggle("visible", j === i));
    indicador.forEach((li, j) => {
      li.classList.toggle("activo", j === i);
      li.classList.toggle("hecho", j < i);
    });
    if (desplazar) window.scrollTo({ top: form.offsetTop - 140, behavior: "smooth" });
  }

  function validar(paso) {
    let ok = true;
    paso.querySelectorAll("input[required]").forEach((inp) => {
      const campo = inp.closest(".campo");
      const vacio = inp.value.trim() === "";
      const fuera = !vacio && !inp.checkValidity();
      campo.classList.toggle("campo-error", vacio || fuera);
      let msg = campo.querySelector(".msg-error");
      if (vacio || fuera) {
        if (!msg) {
          msg = document.createElement("small");
          msg.className = "msg-error";
          campo.appendChild(msg);
        }
        msg.textContent = vacio ? "Falta este dato." : `Debe estar entre ${inp.min} y ${inp.max}.`;
        ok = false;
      } else if (msg) {
        msg.remove();
      }
    });
    if (!ok) paso.querySelector(".campo-error input")?.focus();
    return ok;
  }

  form.querySelectorAll("[data-siguiente]").forEach((b) =>
    b.addEventListener("click", () => validar(pasos[actual]) && mostrar(actual + 1))
  );
  form.querySelectorAll("[data-atras]").forEach((b) =>
    b.addEventListener("click", () => mostrar(actual - 1))
  );
  form.addEventListener("submit", (e) => {
    for (let i = 0; i < pasos.length; i++) {
      if (!validar(pasos[i])) {
        e.preventDefault();
        mostrar(i);
        return;
      }
    }
  });

  // Escalas 0-10: muestra el valor y permite marcar "Sin dato" en las opcionales.
  form.querySelectorAll("[data-escala]").forEach((esc) => {
    const rango = esc.querySelector("input[type=range]");
    const salida = esc.querySelector("output");
    const sinDato = esc.querySelector("[data-sin-dato]");
    const pintar = () => {
      salida.textContent = rango.disabled ? "—" : rango.value;
      const pct = (rango.value / 10) * 100;
      rango.style.setProperty("--pct", rango.disabled ? "0%" : pct + "%");
    };
    rango.addEventListener("input", pintar);
    sinDato?.addEventListener("change", () => {
      rango.disabled = sinDato.checked;
      pintar();
    });
    pintar();
  });

  // Si el servidor devolvió errores, abrir el primer paso con error.
  const conError = pasos.findIndex((p) => p.querySelector(".campo-error"));
  mostrar(conError >= 0 ? conError : 0, false);
})();
