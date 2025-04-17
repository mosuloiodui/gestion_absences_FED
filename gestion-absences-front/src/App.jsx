import { useState } from 'react'
import './App.css'
import ListEmploye from './Comp/Employe/ListEmploye'
import EmployeeForm from './Comp/Employe/AddEmployeForm'
import EditEmployeForm from './Comp/Employe/EditEmployeForm'
import AdminMenu from './Comp/Menu/AdminMenu'
import { createBrowserRouter, RouterProvider, Route } from 'react-router-dom'
import ListDemandes from './Comp/Demandes/ListDemandes'
import AddDemandesForm from './Comp/Demandes/AddDemandesForm'

function App() {
  const [count, setCount] = useState(0)

  const routes = createBrowserRouter([
    { 
      path:"/",
      element: <AdminMenu/>,
      children:[
      {
        path: "/",
        element: <ListEmploye />
      },
      {
        path: "/addEmploye",
        element: <EmployeeForm />
      },
      {
        path: "/editEmploye",
        element: <EditEmployeForm />
      },
      {
        path: "/listDemandes",
        element: <ListDemandes />
      },
      {
        path: "/addDemande",
        element: <AddDemandesForm />
      }
    ]
    }

  ]);

  return (
    <>
      <RouterProvider router={routes}>

      </RouterProvider>
    </>
  )
}

export default App
