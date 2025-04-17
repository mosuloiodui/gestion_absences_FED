import { Outlet } from 'react-router-dom';
import { Link } from 'react-router-dom';


export default function AdminMenu(){

    return(
        <>
        
      <body style={{margin: '25px 50px 75px 100px'}}>
      <ul className="nav nav-pills nav-fill gap-2 p-1 small bg-primary rounded-5 shadow-sm" id="pillNav2" role="tablist"   style={{
        '--bs-nav-link-color': 'var(--bs-white)',
        '--bs-nav-pills-link-active-color': 'var(--bs-primary)',
        '--bs-nav-pills-link-active-bg': 'var(--bs-white)'
      }}>
      <li className="nav-item" role="presentation">
      <button className="btn btn-primary dropdown-toggle rounded-5" type="button" data-bs-toggle="dropdown" aria-expanded="true">
        Gestion des Employés
      </button>
      <ul className="dropdown-menu">
        <li><Link to="/addEmploye" className="dropdown-item">Ajout employé</Link></li>
        <li><Link to="/" className="dropdown-item" >Liste des Employés</Link></li>
      </ul>
      </li>
      <li className="nav-item" role="presentation">
      <button className="btn btn-primary dropdown-toggle rounded-5" type="button" data-bs-toggle="dropdown" aria-expanded="false">
        Gestion des Absences
      </button>
      <ul className="dropdown-menu">
        <li><Link to="/addDemande" className="dropdown-item">Ajout Absence</Link></li>
        <li><Link to= "/listDemandes" className="dropdown-item">Liste des Absences</Link></li>
      </ul>
      </li>
      <li className="nav-item" role="presentation">
      <form   method="post">
        <button className="nav-link" id="home-tab2" data-bs-toggle="tab" type="submit" role="tab" aria-selected="true">Déconnexion</button>
    </form>
        </li>
    </ul>
    </body>
    <Outlet />

    </>
    )
    
    
}