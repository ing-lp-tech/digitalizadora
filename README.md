# Guía para colaboradores — digitalizadora

Este documento explica cómo trabajar en este repositorio y qué información necesita entregar el desarrollador para que el proyecto pueda desplegarse y mantenerse sin depender de una sola persona.

## 1. Flujo de trabajo con Git

- No se sube código directo a la rama `main`. Todo cambio se hace en una rama nueva (por ejemplo `feature/login` o `fix/error-carga`) y se integra a `main` mediante un **pull request**.
- Cada pull request debe tener una descripción breve de qué cambia y por qué.
- Antes de mergear un pull request, debe quedar revisado y aprobado.
- No se suben archivos `.env`, credenciales, claves de API ni carpetas generadas automáticamente (`node_modules`, `dist`, `build`, etc.). Estas exclusiones deben estar en el `.gitignore` desde el primer commit.

## 2. Documentación que el desarrollador debe entregar

Además del código, el proyecto debe incluir (en el `README.md` o en archivos separados):

- **Stack tecnológico**: lenguaje y versión, framework, base de datos y cualquier servicio externo que use el proyecto (por ejemplo Supabase, un servicio de almacenamiento de imágenes, un proveedor de emails, etc.).
- **Variables de entorno**: un archivo `.env.example` que liste todas las variables que necesita el proyecto para funcionar, sin los valores reales (por ejemplo `DATABASE_URL=`, `API_KEY=`).
- **Comandos de instalación y arranque**: los pasos exactos para levantar el proyecto de cero, tanto en modo desarrollo como en modo producción (instalación de dependencias, migraciones de base de datos si las hay, comando para iniciar el servidor).
- **Tipo de hosting requerido**: si la aplicación necesita un servidor corriendo permanentemente (VPS, Render, Railway, etc.), si se puede desplegar como funciones serverless, o si es un frontend estático.

## 3. Integración y despliegue continuo (CI/CD)

Se espera que el proyecto quede configurado con **GitHub Actions** (u otra herramienta equivalente si el desarrollador tiene una justificación técnica para usar otra) de forma que:

- Cuando se aprueba y mergea un pull request a `main`, el sistema automáticamente instale las dependencias, corra las pruebas (si existen) y despliegue la nueva versión al entorno de producción, sin intervención manual.
- Si en algún caso el despliegue automático no es viable (por ejemplo, por restricciones del hosting elegido), el desarrollador debe dejar documentado el procedimiento de despliegue manual paso a paso en el `README.md`, incluyendo qué comandos ejecutar y en qué plataforma.
- El resultado de cada despliegue (si funcionó o falló) debe ser visible desde la pestaña "Actions" del repositorio en GitHub.

## 4. Entornos

Si el proyecto lo justifica, se debe aclarar si existen entornos separados (por ejemplo, "desarrollo/staging" y "producción") y cómo se diferencian las variables de entorno entre uno y otro.

## 5. Accesos y credenciales

- Ninguna credencial (claves de API, contraseñas de base de datos, tokens) debe quedar escrita directamente en el código ni subida al repositorio.
- Las credenciales de servicios externos (hosting, base de datos, etc.) deben quedar registradas en una cuenta de la organización, no en la cuenta personal del desarrollador, para evitar perder el acceso si termina la colaboración.
- Al finalizar el proyecto (o cada hito importante), el desarrollador debe confirmar por escrito qué accesos y cuentas quedaron configurados y quién los administra.